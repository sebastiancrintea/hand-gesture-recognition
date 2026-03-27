import csv
import itertools
import math
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import torch  # noqa: F401 — must import before onnxruntime to load CUDA DLLs
from google.protobuf.json_format import MessageToDict
from scipy.special import softmax

from config import settings
from utils.logger import logger


class HandTracker:
    @staticmethod
    def normalize_landmarks(landmarks: Any, is_left: bool = False) -> list[float]:
        # Translate: wrist (landmark 0) becomes origin
        base_x, base_y = landmarks[0].x, landmarks[0].y
        coords = [[lm.x - base_x, lm.y - base_y] for lm in landmarks]

        # Rotate: align wrist→middle-finger MCP (landmark 9) to point upward
        # In image coords y increases downward, so "up" means negative y direction
        mcp9_x, mcp9_y = coords[9]
        angle = math.atan2(mcp9_x, -mcp9_y)  # angle to rotate so MCP9 is at (0, -dist)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        coords = [
            [x * cos_a + y * sin_a, -x * sin_a + y * cos_a]
            for x, y in coords
        ]

        # Mirror left hand so both hands share the same feature space
        if is_left:
            coords = [[-x, y] for x, y in coords]

        # Scale: divide by wrist→MCP9 distance for anatomically stable normalization
        scale = math.hypot(coords[9][0], coords[9][1])
        if scale > 0:
            coords = [[x / scale, y / scale] for x, y in coords]

        landmark_list = list(itertools.chain.from_iterable(coords))
        return landmark_list

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=settings.MAX_NUM_HANDS,
            model_complexity=0,
            min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.frame_count = 0
        self.last_results = None

        self.point_style = self.mp_draw.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=4
        )
        self.line_style = self.mp_draw.DrawingSpec(
            color=(80, 44, 121), thickness=2, circle_radius=2
        )
        self.bbox_color = (80, 44, 121)
        self.text_color = (255, 255, 255)

        self.labels = []
        try:
            with open(
                "model/keypoint_classifier/keypoint_classifier_label.csv",
                encoding="utf-8-sig",
            ) as f:
                reader = csv.reader(f)
                self.labels = [row[0] for row in reader]
        except Exception as e:
            logger.error(f"Error loading labels: {e}")

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(
                "model/keypoint_classifier/keypoint_classifier.onnx",
                providers=providers,
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            logger.info(
                f"ONNX Inference using providers: {self.ort_session.get_providers()}"
            )
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            self.ort_session = None

    def process(self, frame: np.ndarray) -> Any:

        if self.frame_count % settings.FRAME_SKIP == 0:
            if settings.INFERENCE_WIDTH > 0 and settings.INFERENCE_HEIGHT > 0:
                frame_small = cv2.resize(
                    frame, (settings.INFERENCE_WIDTH, settings.INFERENCE_HEIGHT)
                )
            else:
                frame_small = frame

            img_rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            img_rgb_small.flags.writeable = False

            results = self.hands.process(img_rgb_small)
            img_rgb_small.flags.writeable = True

            if results.multi_hand_landmarks and self.ort_session:
                self.gesture_ids = []
                self.gesture_scores = []
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    is_left = handedness.classification[0].label == "Left"
                    landmark_list = HandTracker.normalize_landmarks(
                        hand_landmarks.landmark, is_left
                    )

                    input_data = np.array([landmark_list], dtype=np.float32)
                    probs = softmax(self.ort_session.run(None, {self.input_name: input_data})[0][0])
                    gesture_id = np.argmax(probs)
                    max_score = probs[gesture_id]

                    self.gesture_ids.append(gesture_id)
                    self.gesture_scores.append(max_score)
            else:
                self.gesture_ids = []
                self.gesture_scores = []

            self.last_results = results

        self.frame_count += 1
        return self.last_results

    def draw(
        self,
        frame: np.ndarray,
        results: Any,
        suppress_gesture_hands: set[int] | None = None,
    ) -> None:

        if not results or not results.multi_hand_landmarks:
            return

        h, w, _ = frame.shape

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.point_style,
                self.line_style,
            )

            landmarks = hand_landmarks.landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
            y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.bbox_color, 2)

            if results.multi_handedness:
                try:
                    handedness_label = MessageToDict(results.multi_handedness[i])[
                        "classification"
                    ][0]["label"]

                    if (
                        hasattr(self, "gesture_ids")
                        and i < len(self.gesture_ids)
                        and (
                            suppress_gesture_hands is None
                            or i not in suppress_gesture_hands
                        )
                    ):
                        gesture_id = self.gesture_ids[i]
                        if gesture_id < len(self.labels):
                            gesture_name = self.labels[gesture_id]
                            confidence = self.gesture_scores[i]

                            label_text = f"{gesture_name} ({confidence * 100:.1f}%)"
                            label_color = (0, 0, 255) if gesture_name == "FingerHeart" else (0, 255, 0)

                            cv2.putText(
                                frame,
                                label_text,
                                (x_min, y_min - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                label_color,
                                2,
                                cv2.LINE_AA,
                            )

                    cv2.putText(
                        frame,
                        handedness_label,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        self.text_color,
                        2,
                    )
                except Exception:
                    pass

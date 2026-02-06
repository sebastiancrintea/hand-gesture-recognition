import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from config import settings
from utils.logger import logger

import torch  # Import torch first to load CUDA DLLs
import onnxruntime as ort
import numpy as np
from scipy.special import softmax
import csv
import itertools


class HandTracker:
    @staticmethod
    def normalize_landmarks(image_width, image_height, landmarks):
        base_x, base_y = landmarks[0].x, landmarks[0].y

        landmark_list = []
        for lm in landmarks:
            landmark_list.append([lm.x - base_x, lm.y - base_y])

        landmark_list = list(itertools.chain.from_iterable(landmark_list))
        max_value = max(list(map(abs, landmark_list)))

        def normalize_(n):
            return n / max_value if max_value > 0 else 0

        landmark_list = list(map(normalize_, landmark_list))
        return landmark_list

    def __init__(self):
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

    def process(self, frame):

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
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    landmark_list = HandTracker.normalize_landmarks(
                        w, h, hand_landmarks.landmark
                    )

                    input_data = np.array([landmark_list], dtype=np.float32)

                    outputs = self.ort_session.run(None, {self.input_name: input_data})

                    output_tensor = outputs[0]

                    probs = softmax(output_tensor[0])
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

    def draw(self, frame, results):

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

                    if hasattr(self, "gesture_ids") and i < len(self.gesture_ids):
                        gesture_id = self.gesture_ids[i]
                        if gesture_id < len(self.labels):
                            gesture_name = self.labels[gesture_id]
                            confidence = self.gesture_scores[i]

                            label_text = f"{gesture_name} ({confidence * 100:.1f}%)"

                            cv2.putText(
                                frame,
                                label_text,
                                (x_min, y_min - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 0),
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

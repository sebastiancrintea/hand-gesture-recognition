import time
from typing import Any

import cv2
import numpy as np

from config import settings
from utils.cvfpscalc import CvFpsCalc
from utils.hand_tracker import HandTracker
from utils.logger import logger
from utils.webcam import WebCamStream

HEART_CONFIDENCE_THRESHOLD = 0.6
HEART_PROXIMITY_PX = 150


def detect_two_hand_heart(
    tracker: HandTracker, results: Any, frame: np.ndarray
) -> tuple[bool, tuple[int, int] | None]:
    """Returns (detected, center_xy) when both hands show HeartHalf close together."""
    if not results or not results.multi_hand_landmarks:
        return False, None
    if len(results.multi_hand_landmarks) != 2 or len(tracker.gesture_ids) != 2:
        return False, None

    try:
        heart_half_id = tracker.labels.index("HeartHalf")
    except ValueError:
        return False, None

    if not all(gid == heart_half_id for gid in tracker.gesture_ids):
        return False, None
    if not all(score >= HEART_CONFIDENCE_THRESHOLD for score in tracker.gesture_scores):
        return False, None

    h, w, _ = frame.shape
    centers = []
    top_ys = []
    for hand_landmarks in results.multi_hand_landmarks:
        lms = hand_landmarks.landmark
        cx = int((min(lm.x for lm in lms) + max(lm.x for lm in lms)) / 2 * w)
        cy = int((min(lm.y for lm in lms) + max(lm.y for lm in lms)) / 2 * h)
        centers.append((cx, cy))
        top_ys.append(int(min(lm.y for lm in lms) * h) - 60)

    dist = (
        (centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2
    ) ** 0.5
    if dist > HEART_PROXIMITY_PX:
        return False, None

    anchor = (
        (centers[0][0] + centers[1][0]) // 2,
        min(top_ys) - 20,
    )
    return True, anchor


def draw_heart_label(frame: np.ndarray, center: tuple[int, int]) -> None:
    text = "Heart <3"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - tw // 2
    y = center[1]
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)


def main() -> None:
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(cv2.getNumThreads(), 4))
        logger.info(
            f"Using OpenCV with optimization: {cv2.useOptimized()} and {cv2.getNumThreads()} threads"
        )
    except Exception:
        logger.warning(
            "OpenCV optimization not available, running with default settings."
        )
        pass

    vs = WebCamStream(src=0).start()
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    tracker = HandTracker()

    fps_display = 0
    last_fps_update = time.time()
    fps_cooldown = 1.0

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)

        results = tracker.process(frame)

        heart_detected, heart_center = detect_two_hand_heart(tracker, results, frame)
        suppress = {0, 1} if heart_detected else None

        tracker.draw(frame, results, suppress_gesture_hands=suppress)

        if heart_detected:
            draw_heart_label(frame, heart_center)

        fps_current = cvFpsCalc.get()

        if time.time() - last_fps_update > fps_cooldown:
            fps_display = int(fps_current)
            last_fps_update = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps_display}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
        )

        if settings.DISPLAY_WIDTH > 0 and settings.DISPLAY_HEIGHT > 0:
            frame_display = cv2.resize(
                frame, (settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT)
            )
        else:
            frame_display = frame

        cv2.imshow("Threaded Ultra-Fast Tracker", frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

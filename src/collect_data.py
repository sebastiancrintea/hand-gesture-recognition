import copy
import csv
import os

import cv2
import numpy as np

from utils.hand_tracker import HandTracker
from utils.logger import logger
from utils.webcam import WebCamStream

LABEL_PATH = "model/keypoint_classifier/keypoint_classifier_label.csv"
CSV_PATH = "model/keypoint_classifier/keypoint.csv"
BURST_SIZE = 100


def load_labels() -> list[str]:
    labels = []
    try:
        with open(LABEL_PATH, encoding="utf-8-sig") as f:
            labels = [row[0] for row in csv.reader(f)]
    except Exception as e:
        logger.error(f"Could not load labels: {e}")
    return labels


def count_samples() -> dict[int, int]:
    counts = {}
    if not os.path.exists(CSV_PATH):
        return counts
    with open(CSV_PATH, newline="") as f:
        for row in csv.reader(f):
            if row:
                cls = int(row[0])
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def logging_csv(class_id: int, landmark_list: list[float]) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([class_id, *landmark_list])


def draw_legend(
    image: np.ndarray,
    labels: list[str],
    sample_counts: dict[int, int],
    active_class: int,
    input_buffer: str,
) -> None:
    x, y = 10, 30
    for class_id, label in enumerate(labels):
        count = sample_counts.get(class_id, 0)
        text = f"[{class_id}] {label} ({count})"
        color = (0, 255, 0) if class_id == active_class else (200, 200, 200)
        cv2.putText(
            image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
        )
        y += 22

    prompt = f"ID> {input_buffer}_"
    cv2.putText(
        image,
        prompt,
        (x, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    os.makedirs("model/keypoint_classifier", exist_ok=True)
    if not os.path.exists(CSV_PATH):
        open(CSV_PATH, "w").close()

    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(cv2.getNumThreads(), 4))
    except Exception:
        pass

    labels = load_labels()
    if not labels:
        logger.error(
            "No labels found. Add gestures to keypoint_classifier_label.csv first."
        )
        return

    logger.info(f"Loaded {len(labels)} gestures: {labels}")
    logger.info(
        "Type a class ID and press Enter to start recording. Press 'q' to quit."
    )

    vs = WebCamStream(src=0).start()
    tracker = HandTracker()

    samples_to_collect = 0
    target_class = -1
    input_buffer = ""

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        results = tracker.process(frame)
        tracker.draw(debug_image, results)

        sample_counts = count_samples()
        draw_legend(debug_image, labels, sample_counts, target_class, input_buffer)

        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"):
            break
        elif key in (8, 127):  # Backspace
            input_buffer = input_buffer[:-1]
        elif 48 <= key <= 57:  # digit
            input_buffer += chr(key)
        elif key == 13:  # Enter
            if input_buffer:
                class_id = int(input_buffer)
                input_buffer = ""
                if class_id < len(labels):
                    target_class = class_id
                    samples_to_collect = BURST_SIZE
                    logger.info(
                        f"Collecting '{labels[target_class]}' (class {target_class})..."
                    )
                else:
                    logger.warning(
                        f"Class {class_id} not in labels (max {len(labels) - 1})"
                    )

        if samples_to_collect > 0:
            if results and results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    is_left = handedness.classification[0].label == "Left"
                    landmark_list = HandTracker.normalize_landmarks(
                        frame.shape[1], frame.shape[0], hand_landmarks.landmark, is_left
                    )
                    logging_csv(target_class, landmark_list)

                samples_to_collect -= 1
                cv2.putText(
                    debug_image,
                    f"COLLECTING '{labels[target_class]}': {samples_to_collect}",
                    (10, debug_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    debug_image,
                    "NO HAND DETECTED",
                    (10, debug_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Data Collector", debug_image)

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

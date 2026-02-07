import cv2
import csv
import copy
from utils.hand_tracker import HandTracker
from utils.webcam import WebCamStream
from utils.logger import logger
import os


def logging_csv(number, landmark_list):
    if 0 <= number <= 9:
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def main():
    os.makedirs("model/keypoint_classifier", exist_ok=True)

    csv_path = "model/keypoint_classifier/keypoint.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline=""):
            pass

    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(cv2.getNumThreads(), 4))
        logger.info(
            f"Using OpenCV with optimization: {cv2.useOptimized()} and {cv2.getNumThreads()} threads"
        )
    except Exception:
        logger.warning("OpenCV optimization not available, running with default settings.")
        pass

    vs = WebCamStream(src=0).start()
    tracker = HandTracker()

    logger.info("Collecting Data...")
    logger.info("Press 0-9 to START recording a burst of samples for that class.")
    logger.info("Press 'q' to quit.")

    samples_to_collect = 0
    target_class = -1

    BURST_SIZE = 100

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        results = tracker.process(frame)
        tracker.draw(debug_image, results)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):  # ESC or q
            break

        if 48 <= key <= 57:  # 0 ~ 9
            target_class = key - 48
            samples_to_collect = BURST_SIZE
            logger.info(f"Starting collection for class {target_class}...")

        if samples_to_collect > 0:
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = HandTracker.normalize_landmarks(
                        frame.shape[1], frame.shape[0], hand_landmarks.landmark
                    )
                    logging_csv(target_class, landmark_list)

                samples_to_collect -= 1
                cv2.putText(
                    debug_image,
                    f"COLLECTING CLASS {target_class}: {samples_to_collect}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    debug_image,
                    "NO HAND DETECTED",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Data Collector", debug_image)

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

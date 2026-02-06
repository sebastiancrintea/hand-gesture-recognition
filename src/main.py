import cv2
import time
from utils.cvfpscalc import CvFpsCalc
from utils.hand_tracker import HandTracker
from utils.webcam import WebCamStream
from config import settings
from utils.logger import logger


def main():
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

        tracker.draw(frame, results)

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

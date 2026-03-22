import threading

import cv2
import numpy as np


class WebCamStream:
    def __init__(self, src: int = 0) -> None:
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self) -> "WebCamStream":
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self) -> None:
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self) -> np.ndarray:
        return self.frame

    def stop(self) -> None:
        self.stopped = True
        self.stream.release()

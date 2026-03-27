class Settings:
    MAX_NUM_HANDS: int = 10
    MIN_DETECTION_CONFIDENCE: float = 0.7
    FRAME_SKIP: int = 4
    BATCH_SIZE: int = 32

    # Resolution for AI Inference (Smaller = Faster, Less Precise)
    # Keep aspect ratio roughly same as webcam (usually 16:9)
    # Set to 0 to use the original camera resolution
    INFERENCE_WIDTH: int = 640
    INFERENCE_HEIGHT: int = 360
    # INFERENCE_WIDTH: int = 0
    # INFERENCE_HEIGHT: int = 0

    # Resolution for Display (Window Size)
    # Set to 0 to use the original camera resolution
    DISPLAY_WIDTH: int = 0
    DISPLAY_HEIGHT: int = 0


settings = Settings()

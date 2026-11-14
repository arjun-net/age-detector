"""Application configuration."""

import os
from pathlib import Path


class Config:
    """Configuration values shared by the application factory."""

    MAX_CONTENT_LENGTH = 8 * 1024 * 1024
    MODEL_NAME = os.getenv("AGE_MODEL_NAME", "buffalo_l")
    MODEL_ROOT = os.getenv("AGE_MODEL_ROOT", str(Path.home() / ".insightface"))
    AGE_MODEL_PATH = os.getenv(
        "AGE_MODEL_PATH",
        str(Path.home() / ".cache" / "age-detector" / "mivolo_v2_age.onnx"),
    )
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

"""Face detection and apparent-age estimation.

Face detection uses InsightFace's RetinaFace detector. Age is estimated with
MiVOLO v2, a model trained specifically for continuous age regression.
"""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

MIVOLO_MODEL_URL = (
    "https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx/"
    "resolve/main/mivolo_v2_age.onnx?download=true"
)
MIVOLO_MODEL_SHA256 = "2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951"
MIVOLO_INPUT_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AgeEstimator:
    """Detect faces and estimate each face's apparent age."""

    def __init__(
        self,
        model_name: str,
        model_root: str,
        age_model_path: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._model_root = model_root
        self._age_model_path = Path(
            age_model_path
            or os.getenv(
                "AGE_MODEL_PATH",
                str(Path.home() / ".cache" / "age-detector" / "mivolo_v2_age.onnx"),
            )
        )
        self._detector: Optional[FaceAnalysis] = None
        self._age_session: Optional[ort.InferenceSession] = None

    def _load_detector(self) -> FaceAnalysis:
        if self._detector is None:
            detector = FaceAnalysis(
                name=self._model_name,
                root=self._model_root,
                allowed_modules=["detection"],
                providers=["CPUExecutionProvider"],
            )
            detector.prepare(ctx_id=0, det_size=(640, 640))
            self._detector = detector
        return self._detector

    def _load_age_model(self) -> ort.InferenceSession:
        if self._age_session is None:
            ensure_model_file(self._age_model_path)
            self._age_session = ort.InferenceSession(
                str(self._age_model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._age_session

    def estimate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Return an apparent-age estimate for every face in an RGB image."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with three channels")

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self._load_detector().get(image_bgr)
        session = self._load_age_model()
        input_name = session.get_inputs()[0].name
        results: List[Dict[str, Any]] = []

        for face in faces:
            crop = crop_face(image_rgb, face.bbox)
            if crop.size == 0:
                continue

            model_input = prepare_mivolo_input(crop)
            age = float(session.run(None, {input_name: model_input})[0].reshape(-1)[0])
            age = max(0.0, min(age, 100.0))
            rounded_age = int(round(age))
            lower = max(0, rounded_age - 5)
            upper = min(100, rounded_age + 5)

            results.append(
                {
                    "age": rounded_age,
                    "age_decimal": round(age, 1),
                    "age_range": f"{lower}–{upper}",
                    "detection_confidence": round(
                        float(getattr(face, "det_score", 0.0)), 3
                    ),
                    "box": [int(value) for value in face.bbox],
                }
            )

        return sorted(
            results, key=lambda result: result["detection_confidence"], reverse=True
        )


def crop_face(image_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop a detected face with a small amount of surrounding context."""
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image_rgb[y1:y2, x1:x2]


def prepare_mivolo_input(face_rgb: np.ndarray) -> np.ndarray:
    """Prepare one face crop for the six-channel MiVOLO ONNX model."""
    face = letterbox(face_rgb, MIVOLO_INPUT_SIZE)
    face = face.astype(np.float32) / 255.0
    face = (face - IMAGENET_MEAN) / IMAGENET_STD
    face = np.transpose(face, (2, 0, 1))

    # MiVOLO accepts face and body crops. A normalized black body crop tells
    # the model that no body information is available.
    black_body = np.broadcast_to(
        -IMAGENET_MEAN[:, None, None], (3, MIVOLO_INPUT_SIZE, MIVOLO_INPUT_SIZE)
    ).astype(np.float32)

    combined = np.concatenate((face, black_body), axis=0)
    return np.expand_dims(combined, axis=0).astype(np.float32)


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize while preserving aspect ratio and pad with black."""
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def ensure_model_file(path: Path) -> None:
    """Download the MiVOLO model atomically and verify its checksum."""
    if path.exists() and file_sha256(path) == MIVOLO_MODEL_SHA256:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        urllib.request.urlretrieve(MIVOLO_MODEL_URL, temporary_path)

    if file_sha256(temporary_path) != MIVOLO_MODEL_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded age model failed its checksum verification")

    temporary_path.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

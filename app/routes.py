"""HTTP routes for the age estimator."""

from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

bp = Blueprint("main", __name__)


@bp.get("/")
def index() -> str:
    return render_template("index.html")


@bp.post("/api/estimate")
def estimate() -> Tuple[Any, int]:
    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({"error": "Choose an image or capture a photo first."}), 400

    try:
        image = Image.open(BytesIO(image_file.read()))
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "The uploaded file is not a readable image."}), 400

    # Guard against decompression bombs before converting to a NumPy array.
    if image.width * image.height > 20_000_000:
        return jsonify({"error": "The image is too large. Use one below 20 megapixels."}), 400

    pixels = np.asarray(image)
    try:
        results = current_app.extensions["age_estimator"].estimate(pixels)
    except Exception:
        current_app.logger.exception("Age estimation failed")
        return jsonify({"error": "Age estimation failed. Check the server log for details."}), 500

    response: Dict[str, Any] = {
        "faces": results,
        "count": len(results),
        "notice": "Estimate only; visible age may differ from actual age.",
    }
    return jsonify(response)

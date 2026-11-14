"""Application factory."""

from flask import Flask

from .age_estimator import AgeEstimator
from .config import Config
from .routes import bp


def create_app(config_object: type[Config] = Config) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(bp)
    app.extensions["age_estimator"] = AgeEstimator(
        model_name=app.config["MODEL_NAME"],
        model_root=app.config["MODEL_ROOT"],
        age_model_path=app.config["AGE_MODEL_PATH"],
    )
    return app

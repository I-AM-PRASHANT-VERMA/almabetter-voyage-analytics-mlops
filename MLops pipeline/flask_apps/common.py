import json

import logging

import warnings

from functools import lru_cache

from pathlib import Path

from flask import jsonify, render_template_string, request

import joblib

import pandas as pd

from sklearn.exceptions import InconsistentVersionWarning

from werkzeug.exceptions import HTTPException


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset" / "travel_capstone"

JOBLIB_DIR = BASE_DIR / "joblib files"

LOGGER = logging.getLogger("voyage_flask_apps")

ERROR_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ service_name }} - {{ status_code }}</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, sans-serif;
            background: #f5f7fb;
            color: #172033;
            margin: 0;
            padding: 2rem;
        }
        .shell {
            max-width: 720px;
            margin: 2rem auto;
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }
        .code {
            color: #1769e0;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        h1 {
            margin: 0.5rem 0 0.75rem;
            font-size: 1.8rem;
        }
        p {
            line-height: 1.6;
            color: #4f5f75;
        }
    </style>
</head>
<body>
    <div class="shell">
        <div class="code">{{ status_code }}</div>
        <h1>{{ service_name }}</h1>
        <p>{{ message }}</p>
    </div>
</body>
</html>
"""


class AssetLoadError(RuntimeError):
    """Raised when a runtime dataset or model artifact is missing or unreadable."""


def ensure_runtime_file(path, label):
    # Fail fast with a clear message before model or dataset loading starts.
    path = Path(path)

    if not path.exists():
        raise AssetLoadError(f"{label} was not found: {path}")

    if path.is_file() and path.stat().st_size == 0:
        raise AssetLoadError(f"{label} is empty: {path}")


def prefers_json_response():
    # The browser landing page is the only HTML route in these Flask demos.
    if request.path != "/":
        return True

    if request.is_json:
        return True

    best_match = request.accept_mimetypes.best_match(
        ["application/json", "text/html"]
    )

    return best_match == "application/json"


def build_error_response(service_name, message, status_code):
    # Keep API routes JSON-first while still returning a readable browser page.
    if prefers_json_response():
        return jsonify(
            {
                "status": "error",
                "app_name": service_name,
                "message": message,
            }
        ), status_code

    return (
        render_template_string(
            ERROR_PAGE_TEMPLATE,
            service_name=service_name,
            message=message,
            status_code=status_code,
        ),
        status_code,
    )


def register_error_handlers(app, service_name):
    # Centralize the common Flask error shapes so all three apps behave the same way.
    @app.errorhandler(AssetLoadError)
    def handle_asset_error(error):
        app.logger.warning("Runtime asset error: %s", error)
        return build_error_response(service_name, str(error), 503)

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        message = error.description or "The request could not be completed."
        return build_error_response(service_name, message, error.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception("Unexpected application error", exc_info=error)
        return build_error_response(
            service_name,
            "The service ran into an unexpected error while processing the request.",
            500,
        )

    return app


def build_health_response(service_name, asset_loader):
    # Health checks should confirm the app can still see the model and dataset it needs.
    try:
        asset_loader()
    except AssetLoadError as error:
        LOGGER.warning("Health check failed for %s: %s", service_name, error)
        return (
            jsonify(
                {
                    "status": "error",
                    "app_name": service_name,
                    "assets_loaded": False,
                    "message": str(error),
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "app_name": service_name,
            "assets_loaded": True,
        }
    )


# The saved gender classifier can warn about sklearn version drift while still loading fine.
def ignore_model_version_warning():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def read_request_data(request):
    # The Flask endpoints accept JSON, form posts, and query-string calls so
    # the browser pages and the API routes can share the same handlers.
    json_payload = request.get_json(silent=True)

    if isinstance(json_payload, dict):
        return json_payload

    form_payload = request.form.to_dict()

    if form_payload:
        return form_payload

    return request.args.to_dict()


def read_positive_int(raw_value, default_value):
    # Most API routes use top_n style inputs, so normalize that parsing in one place.
    try:
        parsed_value = int(raw_value)

    except (TypeError, ValueError):
        return default_value

    if parsed_value > 0:
        return parsed_value

    return default_value


# JSON-serializable records keep Flask responses and Jinja tables consistent.
def dataframe_to_records(dataframe):
    if dataframe.empty:
        return []

    return json.loads(dataframe.to_json(orient="records"))


# Cache model loads because the Flask apps reuse the same saved artifacts on every request.
@lru_cache(maxsize=None)
def load_joblib_file(model_path):
    ignore_model_version_warning()

    ensure_runtime_file(model_path, "Model artifact")

    try:
        return joblib.load(model_path)
    except Exception as error:
        raise AssetLoadError(
            f"Model artifact could not be loaded: {Path(model_path)}"
        ) from error


# Cache CSV loads for the same reason: the source datasets are static while the app is running.
@lru_cache(maxsize=None)
def load_csv_file(csv_path):
    ensure_runtime_file(csv_path, "Dataset file")

    try:
        return pd.read_csv(csv_path)
    except Exception as error:
        raise AssetLoadError(
            f"Dataset file could not be loaded: {Path(csv_path)}"
        ) from error

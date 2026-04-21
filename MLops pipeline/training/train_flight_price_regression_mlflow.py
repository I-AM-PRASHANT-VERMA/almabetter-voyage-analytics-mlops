import argparse

import json

import math

import os

import tempfile

from datetime import datetime

from pathlib import Path

import joblib

import mlflow

import mlflow.sklearn

import pandas as pd

from mlflow.models.signature import infer_signature

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor


# Keep the artifact paths and feature order in one place so the
# training script and the serving apps stay in sync.
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "dataset" / "travel_capstone" / "flights.csv"

DEFAULT_MODEL_PATH = BASE_DIR / "joblib files" / "flight_price_model.joblib"

DEFAULT_METADATA_PATH = BASE_DIR / "joblib files" / "flight_price_model_metadata.json"

DEFAULT_TRACKING_DIR = BASE_DIR / "mlruns"

LOCAL_MLFLOW_TEMP_DIR = BASE_DIR / ".mlflow_temp"

DEFAULT_EXPERIMENT_NAME = "voyage_analytics_flight_price_regression"

MODEL_FEATURE_COLUMNS = [
    "time",
    "year",
    "month",
    "day",
    "from_Brasilia (DF)",
    "from_Campo Grande (MS)",
    "from_Florianopolis (SC)",
    "from_Natal (RN)",
    "from_Recife (PE)",
    "from_Rio de Janeiro (RJ)",
    "from_Salvador (BH)",
    "from_Sao Paulo (SP)",
    "to_Brasilia (DF)",
    "to_Campo Grande (MS)",
    "to_Florianopolis (SC)",
    "to_Natal (RN)",
    "to_Recife (PE)",
    "to_Rio de Janeiro (RJ)",
    "to_Salvador (BH)",
    "to_Sao Paulo (SP)",
    "flightType_firstClass",
    "flightType_premium",
    "agency_FlyingDrops",
    "agency_Rainbow",
]

DEFAULT_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

REQUIRED_DATA_COLUMNS = [
    "date",
    "time",
    "price",
    "from",
    "to",
    "flightType",
    "agency",
]


# Build a local file-based tracking URI from the repo path.
def build_tracking_uri(tracking_dir):
    tracking_dir.mkdir(parents=True, exist_ok=True)

    return tracking_dir.resolve().as_uri()


# MLflow stores each experiment under its id, so this helper keeps the expected
# artifact path easy to rebuild when the experiment metadata needs a refresh.
def build_experiment_artifact_uri(tracking_dir, experiment_id):
    return (tracking_dir.resolve() / str(experiment_id)).as_uri()


# Patch one key in MLflow's local meta.yaml without pulling in a YAML dependency.
def update_local_meta_yaml_value(meta_path, key, value):
    lines = meta_path.read_text(encoding="utf-8").splitlines()

    line_prefix = f"{key}: "

    key_found = False

    for index, line in enumerate(lines):
        if line.startswith(line_prefix):
            lines[index] = f"{line_prefix}{value}"
            key_found = True
            break

    if not key_found:
        lines.append(f"{line_prefix}{value}")

    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Keep the experiment metadata aligned with the current filesystem path before
# a new run starts writing artifacts into the local file store.
def ensure_local_experiment_artifact_path(experiment_name, tracking_dir):
    # Older runs can point to a different path style after switching between
    # Windows and WSL. Refreshing the file-store metadata avoids that drift.
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(f"MLflow experiment could not be loaded: {experiment_name}")

    expected_artifact_uri = build_experiment_artifact_uri(
        tracking_dir=tracking_dir,
        experiment_id=experiment.experiment_id,
    )

    experiment_meta_path = tracking_dir.resolve() / str(experiment.experiment_id) / "meta.yaml"

    if not experiment_meta_path.exists():
        raise FileNotFoundError(
            f"MLflow experiment metadata file was not found: {experiment_meta_path}"
        )

    if experiment.artifact_location != expected_artifact_uri:
        update_local_meta_yaml_value(
            meta_path=experiment_meta_path,
            key="artifact_location",
            value=expected_artifact_uri,
        )
        print("Updated the local MLflow experiment artifact path for this environment.")
        print(f"Previous artifact location: {experiment.artifact_location}")
        print(f"Current artifact location: {expected_artifact_uri}")

    return expected_artifact_uri


# Read the source dataset used by the flight regression workflow.
def load_regression_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if DATA_PATH.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {DATA_PATH}")

    flights_df = pd.read_csv(DATA_PATH)

    missing_columns = [
        column_name
        for column_name in REQUIRED_DATA_COLUMNS
        if column_name not in flights_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return flights_df


# Rebuild the same feature frame layout used by the current saved model.
def prepare_regression_features(flights_df):
    # Reindex back into the saved schema so the refreshed joblib file keeps
    # matching what the Flask and Streamlit apps expect at prediction time.
    working_df = flights_df.copy()

    working_df["date"] = pd.to_datetime(working_df["date"])

    working_df["year"] = working_df["date"].dt.year

    working_df["month"] = working_df["date"].dt.month

    working_df["day"] = working_df["date"].dt.day

    encoded_df = pd.get_dummies(
        working_df[["from", "to", "flightType", "agency"]],
        prefix={
            "from": "from",
            "to": "to",
            "flightType": "flightType",
            "agency": "agency",
        },
    )

    feature_df = pd.concat(
        [working_df[["time", "year", "month", "day"]], encoded_df],
        axis=1,
    )

    feature_df = feature_df.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)

    target_series = working_df["price"].astype(float)

    return feature_df, target_series


# The training flow still uses XGBoost, so wrap the constructor here and keep
# the setup in one place.
def build_model(model_params):
    return XGBRegressor(**model_params)


# Keep the evaluation summary together so MLflow logging and metadata writing
# both use the same rounded values.
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = math.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


# Write the lightweight metadata JSON that the Flask API exposes later.
def save_model_metadata(metadata_path, metadata):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")


# Stage a small JSON file on disk first, then hand it to MLflow as an artifact.
def log_json_artifact(data, staging_dir, file_name, artifact_path):
    staging_dir.mkdir(parents=True, exist_ok=True)

    json_path = staging_dir / file_name

    json_path.write_text(json.dumps(data, indent=4), encoding="utf-8")

    mlflow.log_artifact(str(json_path), artifact_path=artifact_path)


# End-to-end training flow for the flight regression model.
def run_training(args):
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")

    # Local temp folders are kept inside the repo because MLflow artifact
    # logging is more predictable on Windows when it stays off the system temp.
    local_temp_dir = LOCAL_MLFLOW_TEMP_DIR / "tmp"
    local_temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(local_temp_dir)
    os.environ["TEMP"] = str(local_temp_dir)
    os.environ["TMPDIR"] = str(local_temp_dir)
    tempfile.tempdir = str(local_temp_dir)

    tracking_uri = build_tracking_uri(args.tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)

    # This keeps old experiment metadata from pointing at the wrong artifact path
    # after switching between Windows and WSL.
    ensure_local_experiment_artifact_path(
        experiment_name=args.experiment_name,
        tracking_dir=args.tracking_dir,
    )

    artifact_staging_dir = LOCAL_MLFLOW_TEMP_DIR / "artifacts"

    flights_df = load_regression_data()

    # The feature frame is intentionally rebuilt from raw columns so the saved
    # model keeps the same schema the apps already depend on.
    X, y = prepare_regression_features(flights_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model_params = DEFAULT_MODEL_PARAMS.copy()
    model_params["random_state"] = args.random_state

    # Fall back to a timestamped name when no custom MLflow run name is passed in.
    run_name = args.run_name or f"flight_price_xgboost_{datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        mlflow.log_params(model_params)

        mlflow.log_param("feature_count", len(MODEL_FEATURE_COLUMNS))
        mlflow.log_param("date_features", "year_month_day")
        mlflow.log_param("categorical_encoding", "manual_one_hot_reindex")

        mlflow.set_tags(
            {
                "project_name": "voyage_analytics",
                "task_type": "regression",
                "model_type": "XGBRegressor",
                "dataset_name": "travel_capstone/flights.csv",
                "deployment_model": "joblib files/flight_price_model.joblib",
                "workflow_stage": "training",
            }
        )

        # Train on the prepared split, then score the held-out rows before saving anything.
        model = build_model(model_params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        mlflow.log_metrics(metrics)

        log_json_artifact(
            data={
                "feature_columns": MODEL_FEATURE_COLUMNS,
                "numeric_features": ["time", "year", "month", "day"],
                "categorical_source_columns": ["from", "to", "flightType", "agency"],
                "target_column": "price",
                "baseline_categories_not_explicitly_encoded": [
                    "from_Aracaju (SE)",
                    "to_Aracaju (SE)",
                    "flightType_economic",
                    "agency_CloudFy",
                ],
            },
            staging_dir=artifact_staging_dir,
            file_name="preprocessing_summary.json",
            artifact_path="metadata",
        )

        args.model_output.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, args.model_output)

        mlflow.log_artifact(str(args.model_output), artifact_path="joblib_model")

        input_example = X_test.head(5)

        signature = infer_signature(input_example, model.predict(input_example))

        # The sklearn-flavor artifact is useful in MLflow, but the main workflow
        # should still finish if local file permissions block that extra step.
        sklearn_model_logged = True

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

        except (OSError, PermissionError) as error:
            sklearn_model_logged = False
            mlflow.set_tag("sklearn_model_artifact_logged", "false")
            mlflow.set_tag("sklearn_model_artifact_note", "Skipped because of local file permission limits.")
            print("Skipped MLflow sklearn model artifact logging.")
            print(f"Reason: {error}")

        else:
            mlflow.set_tag("sklearn_model_artifact_logged", "true")

        metadata = {
            "model_name": "flight_price_model",
            "task_type": "regression",
            "model_type": "XGBRegressor",
            "dataset_file": str(DATA_PATH.relative_to(BASE_DIR)),
            "model_file": str(args.model_output.relative_to(BASE_DIR)),
            "tracking_uri": tracking_uri,
            "experiment_name": args.experiment_name,
            "run_name": run_name,
            "run_id": run.info.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "feature_count": len(MODEL_FEATURE_COLUMNS),
            "sklearn_model_artifact_logged": sklearn_model_logged,
            "metrics": metrics,
            "model_params": model_params,
        }

        save_model_metadata(args.metadata_output, metadata)

        mlflow.log_artifact(str(args.metadata_output), artifact_path="metadata")

        print("Training completed successfully.")
        print(f"Experiment name: {args.experiment_name}")
        print(f"Run name: {run_name}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Saved model: {args.model_output}")
        print(f"Saved metadata: {args.metadata_output}")
        print(f"Test RMSE: {metrics['rmse']}")
        print(f"Test MAE: {metrics['mae']}")
        print(f"Test R2: {metrics['r2']}")


# Keep the CLI surface small so the same script is easy to call locally,
# from Airflow, from Docker, and from Jenkins.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the flight price regression model and log the run with MLflow."
    )

    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name for the regression model.",
    )

    parser.add_argument(
        "--run-name",
        default="",
        help="Optional MLflow run name.",
    )

    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=DEFAULT_TRACKING_DIR,
        help="Local folder used by MLflow to store runs.",
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for the trained joblib model.",
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Output path for the model metadata JSON file.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size used for evaluation.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for the split and model training.",
    )

    return parser.parse_args()


# Direct script entrypoint for local runs.
if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        run_training(parsed_args)
    except Exception as error:
        print(f"Training failed: {error}")
        raise

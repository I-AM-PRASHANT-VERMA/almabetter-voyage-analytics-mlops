import math

import sys

from pathlib import Path

from functools import lru_cache

from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from flask_apps.common import (
    DATASET_DIR,
    JOBLIB_DIR,
    build_health_response,
    dataframe_to_records,
    load_csv_file,
    load_joblib_file,
    read_positive_int,
    read_request_data,
    register_error_handlers,
)


app = Flask(__name__, template_folder="templates")
register_error_handlers(app, "Gender Classification Flask API")

# Keep the Flask app pointed at the shared classifier artifact and source data.
MODEL_PATH = JOBLIB_DIR / "gender_classifier_best_model.joblib"

USERS_DATA_PATH = DATASET_DIR / "users.csv"


def calculate_decision_strength(raw_score):
    # Translate the raw margin into a friendlier percentage-style signal for the UI and API.
    strength = 50 + (1 - math.exp(-abs(float(raw_score)))) * 49

    return round(strength, 1)


def explain_strength(strength):
    # Short labels make the classifier output easier to explain to non-technical viewers.
    if strength >= 85:
        return "Strong model signal"

    if strength >= 70:
        return "Moderate model signal"

    return "Close decision boundary"


@lru_cache(maxsize=1)
def load_gender_assets():
    # Cache the model and dataset summaries because every route reads from the same bundle.
    model = load_joblib_file(MODEL_PATH)

    users_df = load_csv_file(USERS_DATA_PATH)

    labeled_users_df = users_df[users_df["gender"].isin(["female", "male"])].copy()

    total_records = int(users_df.shape[0])

    labeled_records = int(labeled_users_df.shape[0])

    unlabeled_records = int((users_df["gender"] == "none").sum())

    company_count = int(users_df["company"].nunique())

    gender_distribution_df = (
        users_df["gender"].value_counts().rename_axis("gender").reset_index(name="count")
    )

    company_distribution_df = (
        users_df["company"].value_counts().rename_axis("company").reset_index(name="count")
    )

    sample_names = users_df["name"].dropna().head(30).tolist()

    return {
        "model": model,
        "users_df": users_df,
        "labeled_users_df": labeled_users_df,
        "total_records": total_records,
        "labeled_records": labeled_records,
        "unlabeled_records": unlabeled_records,
        "company_count": company_count,
        "gender_distribution_df": gender_distribution_df,
        "company_distribution_df": company_distribution_df,
        "sample_names": sample_names,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    # The template page is a thin wrapper around the same prediction logic exposed by /predict.
    assets = load_gender_assets()

    default_person_name = "Anita Sharma"

    person_name = str(request.values.get("person_name", default_person_name)).strip()

    sample_name = str(request.values.get("sample_name", "Use typed name")).strip()

    selected_name = person_name if sample_name == "Use typed name" else sample_name

    selected_name = selected_name.strip()

    prediction_result = None

    page_message = ""

    if request.method == "POST":
        if not selected_name:
            page_message = "Please enter a name before running the prediction."

        else:
            # Keep both the predicted label and the decision details for the result panel.
            predicted_gender = str(assets["model"].predict([selected_name])[0])

            raw_score = float(assets["model"].decision_function([selected_name])[0])

            decision_strength = calculate_decision_strength(raw_score)

            prediction_result = {
                "name": selected_name,
                "predicted_gender": predicted_gender,
                "raw_decision_score": round(raw_score, 4),
                "decision_strength_percent": decision_strength,
                "decision_strength_label": explain_strength(decision_strength),
            }

    return render_template(
        "index.html",
        app_name="Gender Prediction By Name",
        model_file=str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        dataset_file=str(USERS_DATA_PATH.relative_to(PROJECT_ROOT)),
        total_records=assets["total_records"],
        labeled_records=assets["labeled_records"],
        unlabeled_records=assets["unlabeled_records"],
        person_name=person_name,
        sample_name=sample_name,
        sample_names=assets["sample_names"],
        prediction_result=prediction_result,
        page_message=page_message,
        gender_distribution=dataframe_to_records(assets["gender_distribution_df"]),
    )


@app.get("/api")
def api_overview():
    # This route gives a compact overview of the available endpoints and sample payload.
    assets = load_gender_assets()

    return jsonify(
        {
            "app_name": "Gender Classification Flask API",
            "model_file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "dataset_file": str(USERS_DATA_PATH.relative_to(PROJECT_ROOT)),
            "available_routes": {
                "GET /": "Open the browser-based Flask page.",
                "GET /api": "Open the JSON overview for this app.",
                "GET /health": "Check whether the API is running.",
                "GET /dataset-summary": "Return dataset totals and distribution tables.",
                "GET /sample-names?top_n=10": "Return sample names from the dataset.",
                "GET or POST /predict": "Return the predicted gender for one name.",
            },
            "sample_payload": {
                "name": "Anita Sharma",
            },
            "summary": {
                "total_records": assets["total_records"],
                "labeled_records": assets["labeled_records"],
                "company_count": assets["company_count"],
            },
        }
    )


@app.get("/health")
def health():
    return build_health_response("Gender Classification Flask API", load_gender_assets)


@app.get("/dataset-summary")
def dataset_summary():
    # Expose dataset totals so the browser page and external clients can show the same summary.
    assets = load_gender_assets()

    return jsonify(
        {
            "total_records": assets["total_records"],
            "labeled_records": assets["labeled_records"],
            "unlabeled_records": assets["unlabeled_records"],
            "company_count": assets["company_count"],
            "gender_distribution": dataframe_to_records(assets["gender_distribution_df"]),
            "company_distribution": dataframe_to_records(assets["company_distribution_df"]),
        }
    )


@app.get("/sample-names")
def sample_names():
    # Limit sample output with top_n so the endpoint stays useful for small demos.
    assets = load_gender_assets()

    top_n = read_positive_int(request.args.get("top_n"), 10)

    return jsonify(
        {
            "top_n": top_n,
            "sample_names": assets["sample_names"][:top_n],
        }
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # The API accepts one name value and returns the same details shown on the browser page.
    assets = load_gender_assets()

    payload = read_request_data(request)

    person_name = str(payload.get("name", "")).strip()

    if not person_name:
        return jsonify({"error": "Please provide a name value."}), 400

    predicted_gender = str(assets["model"].predict([person_name])[0])

    raw_score = float(assets["model"].decision_function([person_name])[0])

    decision_strength = calculate_decision_strength(raw_score)

    strength_label = explain_strength(decision_strength)

    return jsonify(
        {
            "name": person_name,
            "predicted_gender": predicted_gender,
            "raw_decision_score": round(raw_score, 4),
            "decision_strength_percent": decision_strength,
            "decision_strength_label": strength_label,
        }
    )


if __name__ == "__main__":
    # Default local entry point for the gender Flask app.
    app.run(host="0.0.0.0", port=5003, debug=False)

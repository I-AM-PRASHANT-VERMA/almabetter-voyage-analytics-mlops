import sys

import json

from pathlib import Path

from functools import lru_cache

from flask import Flask, jsonify, render_template, request

import pandas as pd


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
    read_request_data,
    register_error_handlers,
)


app = Flask(__name__, template_folder="templates")
register_error_handlers(app, "Flight Price Flask API")

# Keep the Flask service aligned with the artifact locations used by training and Streamlit.
MODEL_PATH = JOBLIB_DIR / "flight_price_model.joblib"

FLIGHTS_DATA_PATH = DATASET_DIR / "flights.csv"

METADATA_PATH = JOBLIB_DIR / "flight_price_model_metadata.json"

# The prediction feature order must match the training script exactly.
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


def build_prediction_input(
    departure_city,
    arrival_city,
    flight_type,
    agency,
    travel_date,
    travel_time,
):
    # Start from the full trained schema, then turn on only the selected route and category flags.
    input_row = {feature_name: 0 for feature_name in MODEL_FEATURE_COLUMNS}

    input_row["time"] = float(travel_time)

    input_row["year"] = int(travel_date.year)

    input_row["month"] = int(travel_date.month)

    input_row["day"] = int(travel_date.day)

    departure_feature_name = f"from_{departure_city}"

    if departure_feature_name in input_row:
        input_row[departure_feature_name] = 1

    arrival_feature_name = f"to_{arrival_city}"

    if arrival_feature_name in input_row:
        input_row[arrival_feature_name] = 1

    input_row["flightType_firstClass"] = 1 if flight_type == "firstClass" else 0

    input_row["flightType_premium"] = 1 if flight_type == "premium" else 0

    input_row["agency_FlyingDrops"] = 1 if agency == "FlyingDrops" else 0

    input_row["agency_Rainbow"] = 1 if agency == "Rainbow" else 0

    # Return a one-row DataFrame in the same order expected by the saved model.
    input_df = pd.DataFrame(
        [[input_row[column_name] for column_name in MODEL_FEATURE_COLUMNS]],
        columns=MODEL_FEATURE_COLUMNS,
    )

    return input_df


def validate_city(city_name, city_options, label):
    # Invalid cities should be rejected clearly instead of falling back to an all-zero route.
    if city_name not in city_options:
        valid_cities = ", ".join(city_options)
        raise ValueError(f"{label} must be one of: {valid_cities}")


def validate_flight_inputs(
    assets,
    departure_city,
    arrival_city,
    flight_type,
    agency,
    travel_time=None,
):
    # Validate every request field before the model receives it.
    validate_city(departure_city, assets["city_options"], "departure_city")
    validate_city(arrival_city, assets["city_options"], "arrival_city")

    if departure_city == arrival_city:
        raise ValueError("departure_city and arrival_city must be different.")

    if flight_type not in assets["flight_type_options"]:
        valid_types = ", ".join(assets["flight_type_options"])
        raise ValueError(f"flight_type must be one of: {valid_types}")

    if agency not in assets["agency_options"]:
        valid_agencies = ", ".join(assets["agency_options"])
        raise ValueError(f"agency must be one of: {valid_agencies}")

    if travel_time is not None and travel_time <= 0:
        raise ValueError("travel_time must be greater than 0.")


def build_route_summary(flights_df):
    # Route summaries make the browser page and API response easier to interpret.
    route_summary_df = (
        flights_df.groupby(["from", "to", "flightType"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_time=("time", "mean"),
            avg_distance=("distance", "mean"),
            trip_count=("travelCode", "count"),
        )
    )

    route_summary_df["avg_price"] = route_summary_df["avg_price"].round(2)

    route_summary_df["avg_time"] = route_summary_df["avg_time"].round(2)

    route_summary_df["avg_distance"] = route_summary_df["avg_distance"].round(2)

    return route_summary_df


def load_model_metadata():
    # Metadata is optional because the model can exist before a tracked training run is logged.
    if not METADATA_PATH.exists():
        return {}

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_flight_assets():
    # Load and cache the shared assets once because every route depends on them.
    model = load_joblib_file(MODEL_PATH)

    flights_df = load_csv_file(FLIGHTS_DATA_PATH)

    route_summary_df = build_route_summary(flights_df)

    city_options = sorted(flights_df["from"].dropna().unique().tolist())

    flight_type_options = sorted(flights_df["flightType"].dropna().unique().tolist())

    agency_options = sorted(flights_df["agency"].dropna().unique().tolist())

    dataset_min_date = str(pd.to_datetime(flights_df["date"]).min().date())

    dataset_max_date = str(pd.to_datetime(flights_df["date"]).max().date())

    average_price = round(float(flights_df["price"].mean()), 2)

    return {
        "model": model,
        "flights_df": flights_df,
        "route_summary_df": route_summary_df,
        "city_options": city_options,
        "flight_type_options": flight_type_options,
        "agency_options": agency_options,
        "dataset_min_date": dataset_min_date,
        "dataset_max_date": dataset_max_date,
        "average_price": average_price,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    # The browser page reuses the same cached assets and prediction logic as the JSON API.
    assets = load_flight_assets()

    default_departure_city = assets["city_options"][0] if assets["city_options"] else ""

    default_arrival_city = (
        assets["city_options"][1]
        if len(assets["city_options"]) > 1
        else default_departure_city
    )

    default_flight_type = (
        assets["flight_type_options"][0] if assets["flight_type_options"] else ""
    )

    default_agency = assets["agency_options"][0] if assets["agency_options"] else ""

    departure_city = str(
        request.values.get("departure_city", default_departure_city)
    ).strip()

    arrival_city = str(
        request.values.get("arrival_city", default_arrival_city)
    ).strip()

    flight_type = str(request.values.get("flight_type", default_flight_type)).strip()

    agency = str(request.values.get("agency", default_agency)).strip()

    travel_date_text = str(
        request.values.get("travel_date", assets["dataset_min_date"])
    ).strip()

    default_travel_time = round(float(assets["flights_df"]["time"].median()), 2)

    travel_time_text = request.values.get("travel_time", default_travel_time)

    if departure_city not in assets["city_options"]:
        departure_city = default_departure_city

    if arrival_city not in assets["city_options"]:
        arrival_city = default_arrival_city

    if flight_type not in assets["flight_type_options"]:
        flight_type = default_flight_type

    if agency not in assets["agency_options"]:
        agency = default_agency

    try:
        travel_time = float(travel_time_text)

    except (TypeError, ValueError):
        travel_time = default_travel_time

    selected_route_df = assets["route_summary_df"][
        (assets["route_summary_df"]["from"] == departure_city)
        & (assets["route_summary_df"]["to"] == arrival_city)
        & (assets["route_summary_df"]["flightType"] == flight_type)
    ]

    predicted_price = None

    model_input_preview = []

    page_error = ""

    if request.method == "POST":
        try:
            # Parse the submitted values into the model-ready feature row used for prediction.
            travel_date = pd.to_datetime(travel_date_text).date()

            prediction_input_df = build_prediction_input(
                departure_city=departure_city,
                arrival_city=arrival_city,
                flight_type=flight_type,
                agency=agency,
                travel_date=travel_date,
                travel_time=travel_time,
            )

            predicted_price = round(
                float(assets["model"].predict(prediction_input_df)[0]), 2
            )

            model_input_preview = dataframe_to_records(prediction_input_df)

            travel_date_text = str(travel_date)

        except Exception:
            page_error = (
                "The app could not create a prediction from the submitted values. "
                "Please check the date and travel time fields."
            )

    return render_template(
        "index.html",
        app_name="Flight Price Prediction",
        model_file=str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        dataset_file=str(FLIGHTS_DATA_PATH.relative_to(PROJECT_ROOT)),
        record_count=int(assets["flights_df"].shape[0]),
        route_count=int(assets["route_summary_df"].shape[0]),
        average_price=assets["average_price"],
        city_options=assets["city_options"],
        flight_type_options=assets["flight_type_options"],
        agency_options=assets["agency_options"],
        dataset_min_date=assets["dataset_min_date"],
        dataset_max_date=assets["dataset_max_date"],
        departure_city=departure_city,
        arrival_city=arrival_city,
        flight_type=flight_type,
        agency=agency,
        travel_date=travel_date_text,
        travel_time=travel_time,
        predicted_price=predicted_price,
        page_error=page_error,
        route_summary=dataframe_to_records(selected_route_df),
    )


@app.get("/api")
def api_overview():
    # This route doubles as a quick reference page for anyone testing the API manually.
    assets = load_flight_assets()

    return jsonify(
        {
            "app_name": "Flight Price Flask API",
            "model_file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "dataset_file": str(FLIGHTS_DATA_PATH.relative_to(PROJECT_ROOT)),
            "available_routes": {
                "GET /": "Open the browser-based Flask page.",
                "GET /api": "Open the JSON overview for this app.",
                "GET /health": "Check whether the API is running.",
                "GET /model-info": "Return metadata for the saved regression model when available.",
                "GET /metadata": "Return the dropdown options and dataset date range.",
                "GET or POST /route-summary": "Return the historical route summary for one route and cabin type.",
                "GET or POST /predict": "Return the predicted flight price for one request payload.",
            },
            "sample_payload": {
                "departure_city": "Recife (PE)",
                "arrival_city": "Florianopolis (SC)",
                "flight_type": "firstClass",
                "agency": "FlyingDrops",
                "travel_date": assets["dataset_min_date"],
                "travel_time": 1.76,
            },
            "summary": {
                "record_count": int(assets["flights_df"].shape[0]),
                "route_count": int(assets["route_summary_df"].shape[0]),
                "average_price": assets["average_price"],
            },
        }
    )


@app.get("/health")
def health():
    return build_health_response("Flight Price Flask API", load_flight_assets)


@app.get("/model-info")
def model_info():
    # Expose MLflow-linked metadata when it exists so the UI can show the latest run details.
    model_metadata = load_model_metadata()

    if not model_metadata:
        return jsonify(
            {
                "message": "Model metadata is not available yet. Run the MLflow training script first.",
                "metadata_file": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
            }
        )

    return jsonify(model_metadata)


@app.get("/metadata")
def metadata():
    # This keeps dropdown options and date limits available to external clients.
    assets = load_flight_assets()

    return jsonify(
        {
            "city_options": assets["city_options"],
            "flight_type_options": assets["flight_type_options"],
            "agency_options": assets["agency_options"],
            "dataset_min_date": assets["dataset_min_date"],
            "dataset_max_date": assets["dataset_max_date"],
        }
    )


@app.route("/route-summary", methods=["GET", "POST"])
def route_summary():
    # Accept either JSON, form data, or query params through the shared request reader.
    assets = load_flight_assets()

    payload = read_request_data(request)

    departure_city = str(payload.get("departure_city", "")).strip()

    arrival_city = str(payload.get("arrival_city", "")).strip()

    flight_type = str(payload.get("flight_type", "")).strip()

    if not departure_city or not arrival_city or not flight_type:
        return jsonify({"error": "Please provide departure_city, arrival_city, and flight_type."}), 400

    try:
        validate_city(departure_city, assets["city_options"], "departure_city")
        validate_city(arrival_city, assets["city_options"], "arrival_city")

        if departure_city == arrival_city:
            raise ValueError("departure_city and arrival_city must be different.")

        if flight_type not in assets["flight_type_options"]:
            valid_types = ", ".join(assets["flight_type_options"])
            raise ValueError(f"flight_type must be one of: {valid_types}")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    selected_route_df = assets["route_summary_df"][
        (assets["route_summary_df"]["from"] == departure_city)
        & (assets["route_summary_df"]["to"] == arrival_city)
        & (assets["route_summary_df"]["flightType"] == flight_type)
    ]

    if selected_route_df.empty:
        return jsonify({"error": "No historical summary was found for the selected route and flight type."}), 404

    return jsonify(
        {
            "route_summary": dataframe_to_records(selected_route_df),
        }
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # The prediction endpoint mirrors the same inputs used by the browser and Streamlit versions.
    assets = load_flight_assets()

    payload = read_request_data(request)

    departure_city = str(payload.get("departure_city", "")).strip()

    arrival_city = str(payload.get("arrival_city", "")).strip()

    flight_type = str(payload.get("flight_type", "")).strip()

    agency = str(payload.get("agency", "")).strip()

    travel_date_text = str(payload.get("travel_date", "")).strip()

    travel_time_text = payload.get("travel_time")

    if not all([departure_city, arrival_city, flight_type, agency, travel_date_text]):
        return jsonify({"error": "Please provide departure_city, arrival_city, flight_type, agency, and travel_date."}), 400

    try:
        travel_date = pd.to_datetime(travel_date_text).date()

    except Exception:
        return jsonify({"error": "Please provide travel_date in a valid date format such as 2026-04-11."}), 400

    try:
        travel_time = float(travel_time_text)

    except (TypeError, ValueError):
        return jsonify({"error": "Please provide travel_time as a numeric value."}), 400

    try:
        validate_flight_inputs(
            assets=assets,
            departure_city=departure_city,
            arrival_city=arrival_city,
            flight_type=flight_type,
            agency=agency,
            travel_time=travel_time,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    prediction_input_df = build_prediction_input(
        departure_city=departure_city,
        arrival_city=arrival_city,
        flight_type=flight_type,
        agency=agency,
        travel_date=travel_date,
        travel_time=travel_time,
    )

    predicted_price = float(assets["model"].predict(prediction_input_df)[0])

    # Include the matching historical route summary so the predicted price has some context.
    selected_route_df = assets["route_summary_df"][
        (assets["route_summary_df"]["from"] == departure_city)
        & (assets["route_summary_df"]["to"] == arrival_city)
        & (assets["route_summary_df"]["flightType"] == flight_type)
    ]

    return jsonify(
        {
            "request": {
                "departure_city": departure_city,
                "arrival_city": arrival_city,
                "flight_type": flight_type,
                "agency": agency,
                "travel_date": str(travel_date),
                "travel_time": travel_time,
            },
            "predicted_price": round(predicted_price, 2),
            "overall_average_price": assets["average_price"],
            "route_summary": dataframe_to_records(selected_route_df),
            "model_input_preview": dataframe_to_records(prediction_input_df),
        }
    )


if __name__ == "__main__":
    # Default local entry point for running the API outside Docker or Kubernetes.
    app.run(host="0.0.0.0", port=5002, debug=False)

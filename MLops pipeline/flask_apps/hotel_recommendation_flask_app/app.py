import sys

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
    read_positive_int,
    read_request_data,
    register_error_handlers,
)


app = Flask(__name__, template_folder="templates")
register_error_handlers(app, "Hotel Recommendation Flask API")

# Keep the hotel app aligned with the shared recommender bundle and hotel dataset.
MODEL_PATH = JOBLIB_DIR / "hotel_recommender_simple.joblib"

HOTELS_DATA_PATH = DATASET_DIR / "hotels.csv"


def build_hotel_summary(hotels_df):
    # Roll booking rows up to one hotel-level summary used across the page and API routes.
    summary_df = (
        hotels_df.groupby("name", as_index=False)
        .agg(
            place=("place", "first"),
            avg_price=("price", "mean"),
            avg_total=("total", "mean"),
            avg_days=("days", "mean"),
            booking_count=("travelCode", "count"),
        )
    )

    summary_df["avg_price"] = summary_df["avg_price"].round(2)

    summary_df["avg_total"] = summary_df["avg_total"].round(2)

    summary_df["avg_days"] = summary_df["avg_days"].round(1)

    return summary_df


def get_popular_hotels(model_bundle, hotel_summary_df):
    # Merge the saved popularity ranking with the descriptive hotel summary fields.
    popular_df = model_bundle["popular_hotels"].copy()

    popular_df = popular_df.merge(hotel_summary_df, on="name", how="left")

    popular_df["avg_rating"] = popular_df["avg_rating"].round(2)

    return popular_df


def get_similar_hotels(model_bundle, hotel_summary_df, selected_hotel, top_n):
    # Similar-hotel results combine the saved content and collaborative similarity signals.
    hotel_index = model_bundle["hotel_index"]

    if selected_hotel not in hotel_index.index:
        return pd.DataFrame()

    selected_position = hotel_index[selected_hotel]

    content_similarity = model_bundle["content_similarity"]

    item_similarity_df = model_bundle["item_similarity_df"]

    hotel_names = list(item_similarity_df.columns)

    recommendation_rows = []

    for hotel_name in hotel_names:
        if hotel_name == selected_hotel:
            continue

        candidate_position = hotel_index[hotel_name]

        content_score = float(content_similarity[selected_position][candidate_position])

        collaborative_score = float(item_similarity_df.loc[selected_hotel, hotel_name])

        combined_score = round((content_score + collaborative_score) / 2, 4)

        recommendation_rows.append(
            {
                "name": hotel_name,
                "content_score": round(content_score, 4),
                "collaborative_score": round(collaborative_score, 4),
                "combined_score": combined_score,
            }
        )

    recommendations_df = pd.DataFrame(recommendation_rows)

    if recommendations_df.empty:
        return recommendations_df

    recommendations_df = recommendations_df.sort_values(
        by="combined_score", ascending=False
    ).head(top_n)

    recommendations_df = recommendations_df.merge(
        hotel_summary_df, on="name", how="left"
    )

    return recommendations_df


def get_user_recommendations(model_bundle, hotel_summary_df, selected_user, top_n):
    # Recommend unseen hotels by weighting each candidate against the user's current interaction history.
    user_item_matrix = model_bundle["user_item_matrix"]

    item_similarity_df = model_bundle["item_similarity_df"]

    if selected_user not in user_item_matrix.index:
        return pd.DataFrame(), pd.DataFrame()

    user_ratings = user_item_matrix.loc[selected_user]

    visited_hotels = user_ratings[user_ratings > 0].sort_values(ascending=False)

    score_rows = []

    for candidate_hotel in user_item_matrix.columns:
        if user_ratings[candidate_hotel] > 0:
            continue

        similarity_vector = item_similarity_df[candidate_hotel]

        weighted_scores = similarity_vector * user_ratings

        numerator = weighted_scores.sum()

        denominator = similarity_vector.abs().sum()

        predicted_score = 0.0 if denominator == 0 else float(numerator / denominator)

        score_rows.append(
            {
                "name": candidate_hotel,
                "predicted_score": round(predicted_score, 4),
            }
        )

    recommendations_df = pd.DataFrame(score_rows)

    history_df = visited_hotels.reset_index()

    history_df.columns = ["name", "interaction_score"]

    history_df = history_df.merge(hotel_summary_df, on="name", how="left")

    if recommendations_df.empty:
        return recommendations_df, history_df

    recommendations_df = recommendations_df.sort_values(
        by="predicted_score", ascending=False
    ).head(top_n)

    recommendations_df = recommendations_df.merge(
        hotel_summary_df, on="name", how="left"
    )

    return recommendations_df, history_df


@lru_cache(maxsize=1)
def load_hotel_assets():
    # Load the saved recommender and all shared summary tables only once.
    model_bundle = load_joblib_file(MODEL_PATH)

    hotels_df = load_csv_file(HOTELS_DATA_PATH)

    hotel_summary_df = build_hotel_summary(hotels_df)

    popular_hotels_df = get_popular_hotels(model_bundle, hotel_summary_df)

    user_item_matrix_df = model_bundle["user_item_matrix"]

    user_seen_counts = (user_item_matrix_df > 0).sum(axis=1)

    eligible_users = user_seen_counts[
        user_seen_counts < user_item_matrix_df.shape[1]
    ].index.tolist()

    return {
        "model_bundle": model_bundle,
        "hotels_df": hotels_df,
        "hotel_summary_df": hotel_summary_df,
        "popular_hotels_df": popular_hotels_df,
        "user_item_matrix_df": user_item_matrix_df,
        "user_seen_counts": user_seen_counts,
        "eligible_users": eligible_users,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    # The browser page gathers all three recommendation views from the same cached asset bundle.
    assets = load_hotel_assets()

    hotel_options = sorted(assets["model_bundle"]["hotel_index"].index.tolist())

    user_options = sorted(assets["user_item_matrix_df"].index.tolist())

    default_hotel = hotel_options[0] if hotel_options else ""

    default_user = (
        assets["eligible_users"][0]
        if assets["eligible_users"]
        else (user_options[0] if user_options else "")
    )

    popular_top_n = read_positive_int(request.values.get("popular_top_n"), 5)

    selected_hotel = str(request.values.get("selected_hotel", default_hotel)).strip()

    similar_top_n = read_positive_int(request.values.get("similar_top_n"), 5)

    selected_user = str(request.values.get("selected_user", default_user)).strip()

    user_top_n = read_positive_int(request.values.get("user_top_n"), 5)

    if selected_hotel not in hotel_options:
        selected_hotel = default_hotel

    if selected_user not in user_options:
        selected_user = default_user

    popular_hotels_df = assets["popular_hotels_df"].head(popular_top_n)

    similar_hotels_df = get_similar_hotels(
        model_bundle=assets["model_bundle"],
        hotel_summary_df=assets["hotel_summary_df"],
        selected_hotel=selected_hotel,
        top_n=similar_top_n,
    )

    user_recommendations_df, user_history_df = get_user_recommendations(
        model_bundle=assets["model_bundle"],
        hotel_summary_df=assets["hotel_summary_df"],
        selected_user=selected_user,
        top_n=user_top_n,
    )

    selected_hotel_details = assets["hotel_summary_df"][
        assets["hotel_summary_df"]["name"] == selected_hotel
    ]

    selected_hotel_details = (
        selected_hotel_details.iloc[0].to_dict()
        if not selected_hotel_details.empty
        else None
    )

    seen_hotel_count = (
        int(assets["user_seen_counts"][selected_user])
        if selected_user in assets["user_seen_counts"].index
        else 0
    )

    return render_template(
        "index.html",
        app_name="Hotel Recommendation System",
        model_file=str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        dataset_file=str(HOTELS_DATA_PATH.relative_to(PROJECT_ROOT)),
        hotel_count=int(assets["hotel_summary_df"].shape[0]),
        user_count=int(assets["user_item_matrix_df"].shape[0]),
        eligible_user_count=int(len(assets["eligible_users"])),
        hotel_options=hotel_options,
        user_options=user_options,
        selected_hotel=selected_hotel,
        selected_user=selected_user,
        selected_hotel_details=selected_hotel_details,
        popular_top_n=popular_top_n,
        similar_top_n=similar_top_n,
        user_top_n=user_top_n,
        seen_hotel_count=seen_hotel_count,
        popular_hotels=dataframe_to_records(popular_hotels_df),
        similar_hotels=dataframe_to_records(similar_hotels_df),
        recommended_hotels=dataframe_to_records(user_recommendations_df),
        user_history=dataframe_to_records(user_history_df),
    )


@app.get("/api")
def api_overview():
    # Provide a quick reference for anyone exploring the hotel endpoints manually.
    assets = load_hotel_assets()

    return jsonify(
        {
            "app_name": "Hotel Recommendation Flask API",
            "model_file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "dataset_file": str(HOTELS_DATA_PATH.relative_to(PROJECT_ROOT)),
            "available_routes": {
                "GET /": "Open the browser-based Flask page.",
                "GET /api": "Open the JSON overview for this app.",
                "GET /health": "Check whether the API is running.",
                "GET /popular-hotels?top_n=5": "Return the most popular hotels from the saved model.",
                "GET or POST /similar-hotels": "Return hotels similar to a selected hotel.",
                "GET or POST /user-recommendations": "Return personalized hotel suggestions for a selected user.",
            },
            "sample_payloads": {
                "similar_hotels": {
                    "hotel_name": "Hotel A",
                    "top_n": 5,
                },
                "user_recommendations": {
                    "user_code": "0",
                    "top_n": 5,
                },
            },
            "summary": {
                "hotel_count": int(assets["hotel_summary_df"].shape[0]),
                "user_count": int(assets["user_item_matrix_df"].shape[0]),
                "eligible_user_count": int(len(assets["eligible_users"])),
            },
        }
    )


@app.get("/health")
def health():
    return build_health_response("Hotel Recommendation Flask API", load_hotel_assets)


@app.get("/popular-hotels")
def popular_hotels():
    # Return only the requested slice from the saved popularity table.
    assets = load_hotel_assets()

    top_n = read_positive_int(request.args.get("top_n"), 5)

    popular_hotels_df = assets["popular_hotels_df"].head(top_n)

    return jsonify(
        {
            "top_n": top_n,
            "results": dataframe_to_records(popular_hotels_df),
        }
    )


@app.route("/similar-hotels", methods=["GET", "POST"])
def similar_hotels():
    # Allow similar-hotel lookups from either query params, forms, or JSON payloads.
    assets = load_hotel_assets()

    payload = read_request_data(request)

    selected_hotel = str(payload.get("hotel_name", "")).strip()

    top_n = read_positive_int(payload.get("top_n"), 5)

    if not selected_hotel:
        return jsonify({"error": "Please provide a hotel_name value."}), 400

    similar_hotels_df = get_similar_hotels(
        model_bundle=assets["model_bundle"],
        hotel_summary_df=assets["hotel_summary_df"],
        selected_hotel=selected_hotel,
        top_n=top_n,
    )

    if similar_hotels_df.empty and selected_hotel not in assets["model_bundle"]["hotel_index"].index:
        return jsonify({"error": f"The hotel '{selected_hotel}' was not found."}), 404

    return jsonify(
        {
            "hotel_name": selected_hotel,
            "top_n": top_n,
            "results": dataframe_to_records(similar_hotels_df),
        }
    )


@app.route("/user-recommendations", methods=["GET", "POST"])
def user_recommendations():
    # Personalized results are returned together with the user's seen history for context.
    assets = load_hotel_assets()

    payload = read_request_data(request)

    selected_user = str(payload.get("user_code", "")).strip()

    top_n = read_positive_int(payload.get("top_n"), 5)

    if not selected_user:
        return jsonify({"error": "Please provide a user_code value."}), 400

    recommendations_df, history_df = get_user_recommendations(
        model_bundle=assets["model_bundle"],
        hotel_summary_df=assets["hotel_summary_df"],
        selected_user=selected_user,
        top_n=top_n,
    )

    if selected_user not in assets["user_item_matrix_df"].index:
        return jsonify({"error": f"The user_code '{selected_user}' was not found."}), 404

    seen_hotels = int(assets["user_seen_counts"][selected_user])

    return jsonify(
        {
            "user_code": selected_user,
            "top_n": top_n,
            "already_seen_hotel_count": seen_hotels,
            "recommended_hotels": dataframe_to_records(recommendations_df),
            "user_history": dataframe_to_records(history_df),
        }
    )


if __name__ == "__main__":
    # Default local entry point for the hotel Flask app.
    app.run(host="0.0.0.0", port=5001, debug=False)

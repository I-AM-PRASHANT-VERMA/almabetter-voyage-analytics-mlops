import os

from pathlib import Path

import joblib

import pandas as pd

import requests

import streamlit as st


st.set_page_config(page_title="Flight Price Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

# Keep the Streamlit app on the same saved artifact used by the Flask API.
MODEL_PATH = BASE_DIR / "joblib files" / "flight_price_model.joblib"

# The dataset is reused for route summaries, date limits, and dashboard metrics.
FLIGHTS_DATA_PATH = BASE_DIR / "dataset" / "travel_capstone" / "flights.csv"

# When this value is present, the UI routes prediction requests through Flask instead of the local model.
FLIGHT_PRICE_API_URL = os.getenv("FLIGHT_PRICE_API_URL", "").strip().rstrip("/")

# This fixed column order has to stay aligned with the training script and Flask API input builder.
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


@st.cache_resource
def load_model():
    # Model loading is cached once so a rerun does not hit disk again.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact was not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_flights_data():
    # The source CSV is static for the session, so cache it with the rest of the page state.
    if not FLIGHTS_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file was not found: {FLIGHTS_DATA_PATH}")

    return pd.read_csv(FLIGHTS_DATA_PATH)


def use_flight_api():
    # A simple environment flag is enough to switch between local and API-backed predictions.
    return bool(FLIGHT_PRICE_API_URL)


@st.cache_data(ttl=300)
def load_api_model_info():
    # This call is only for display details in the sidebar, so an empty response is fine.
    if not use_flight_api():
        return {}

    try:
        response = requests.get(f"{FLIGHT_PRICE_API_URL}/model-info", timeout=10)

        if not response.ok:
            return {}

        return response.json()

    except requests.RequestException:
        return {}


def predict_with_api(
    departure_city,
    arrival_city,
    flight_type,
    agency,
    travel_date,
    travel_time,
):
    # Keep the payload keys aligned with the Flask /predict contract.
    payload = {
        "departure_city": departure_city,
        "arrival_city": arrival_city,
        "flight_type": flight_type,
        "agency": agency,
        "travel_date": str(travel_date),
        "travel_time": float(travel_time),
    }

    response = requests.post(
        f"{FLIGHT_PRICE_API_URL}/predict",
        json=payload,
        timeout=30,
    )

    if not response.ok:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}

        error_message = error_payload.get("error") or error_payload.get("message")

        if not error_message:
            error_message = (
                f"Prediction request failed with status {response.status_code}."
            )

        raise RuntimeError(error_message)

    return response.json()


def format_currency(value):
    # Use the same currency formatting everywhere the fare is shown in the UI.
    return f"Rs. {value:,.2f}"


def get_selected_theme():
    # Store the last theme choice in session state so it survives widget reruns.
    if "flight_theme_mode_v2" not in st.session_state:
        st.session_state.flight_theme_mode_v2 = "System"

    return st.session_state.flight_theme_mode_v2


def apply_theme_css(selected_theme):
    # The theme switch only swaps CSS variables, so the rest of the layout can stay unchanged.
    if selected_theme == "Light":
        theme_variables = """
        :root {
            --bg: #edf3fb;
            --bg-soft: #f7fbff;
            --surface: #ffffff;
            --surface-2: #f8fbff;
            --text: #0f172a;
            --muted: #526277;
            --border: #dbe5f0;
            --primary: #1769e0;
            --primary-soft: rgba(23, 105, 224, 0.10);
            --accent: #0f766e;
            --shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
            --hero-start: #ffffff;
            --hero-end: #e8f1ff;
        }
        """

    elif selected_theme == "Dark":
        theme_variables = """
        :root {
            --bg: #07111f;
            --bg-soft: #0c1930;
            --surface: #0f1b30;
            --surface-2: #13233d;
            --text: #ecf4ff;
            --muted: #9fb2cc;
            --border: #22395f;
            --primary: #66c6ff;
            --primary-soft: rgba(102, 198, 255, 0.12);
            --accent: #34d399;
            --shadow: 0 18px 36px rgba(2, 8, 23, 0.34);
            --hero-start: #13223d;
            --hero-end: #0f1a30;
        }
        """

    else:
        theme_variables = """
        @media (prefers-color-scheme: light) {
            :root {
                --bg: #edf3fb;
                --bg-soft: #f7fbff;
                --surface: #ffffff;
                --surface-2: #f8fbff;
                --text: #0f172a;
                --muted: #526277;
                --border: #dbe5f0;
                --primary: #1769e0;
                --primary-soft: rgba(23, 105, 224, 0.10);
                --accent: #0f766e;
                --shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
                --hero-start: #ffffff;
                --hero-end: #e8f1ff;
            }
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #07111f;
                --bg-soft: #0c1930;
                --surface: #0f1b30;
                --surface-2: #13233d;
                --text: #ecf4ff;
                --muted: #9fb2cc;
                --border: #22395f;
                --primary: #66c6ff;
                --primary-soft: rgba(102, 198, 255, 0.12);
                --accent: #34d399;
                --shadow: 0 18px 36px rgba(2, 8, 23, 0.34);
                --hero-start: #13223d;
                --hero-end: #0f1a30;
            }
        }
        """

    css_text = f"""
    <style>
    {theme_variables}

    #MainMenu {{
        visibility: hidden;
    }}
    footer {{
        visibility: hidden;
    }}
    [data-testid="stHeader"] {{
        display: none;
    }}
    [data-testid="stToolbar"] {{
        display: none;
    }}
    [data-testid="stDecoration"] {{
        display: none;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
        border-right: 1px solid var(--border);
    }}
    .stApp {{
        background:
            radial-gradient(circle at top left, var(--bg-soft) 0%, var(--bg) 38%, var(--bg) 100%);
        color: var(--text);
    }}
    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}
    .block-container {{
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div {{
        color: inherit;
    }}
    [data-testid="stMarkdownContainer"] p {{
        color: var(--muted);
    }}
    [data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 0.95rem 1rem;
        box-shadow: var(--shadow);
    }}
    [data-testid="stMetric"] label {{
        color: var(--muted) !important;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--text) !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 0.45rem;
    }}
    button[data-baseweb="tab"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--muted);
        padding: 0.45rem 1rem;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: var(--primary-soft);
        border-color: var(--primary);
        color: var(--primary);
    }}
    div[data-baseweb="select"] > div {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 16px !important;
        min-height: 3rem !important;
        box-shadow: none !important;
    }}
    div[data-baseweb="input"] > div {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }}
    .stDateInput input {{
        color: var(--text) !important;
    }}
    .stAlert {{
        border-radius: 18px;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 18px;
        overflow: hidden;
    }}
    .hero-card {{
        background:
            radial-gradient(circle at top right, rgba(249, 179, 107, 0.34), transparent 18rem),
            linear-gradient(135deg, #0f172a 0%, #155e75 58%, #d97745 125%);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 2.1rem;
        box-shadow: var(--shadow);
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        color: #fff7ed;
        padding: 0.38rem 0.82rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.9rem;
    }}
    .hero-title {{
        font-size: 3rem;
        line-height: 1.02;
        font-weight: 800;
        color: #fffaf0;
        margin-bottom: 0.8rem;
        letter-spacing: -0.04em;
    }}
    .hero-copy {{
        max-width: 820px;
        font-size: 1.05rem;
        color: rgba(255, 250, 240, 0.84);
    }}
    .panel-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.2rem 1.25rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.9rem;
    }}
    div[data-testid="stForm"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.15rem;
        box-shadow: var(--shadow);
    }}
    .stButton > button,
    div[data-testid="stFormSubmitButton"] button {{
        min-height: 3rem;
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #0f766e 0%, #1769e0 100%);
        color: #ffffff;
        font-weight: 800;
        box-shadow: 0 16px 34px rgba(15, 118, 110, 0.24);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }}
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {{
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 20px 42px rgba(15, 118, 110, 0.30);
    }}
    .section-label {{
        font-size: 1.15rem;
        font-weight: 750;
        color: var(--text);
        margin-bottom: 0.3rem;
    }}
    .section-copy {{
        color: var(--muted);
        font-size: 0.96rem;
        margin-bottom: 0.15rem;
    }}
    .result-shell {{
        background: linear-gradient(145deg, var(--surface) 0%, var(--surface-2) 100%);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.35rem;
        box-shadow: var(--shadow);
    }}
    .result-kicker {{
        color: var(--muted);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
    }}
    .result-price {{
        color: var(--text);
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.35rem;
    }}
    .result-note {{
        color: var(--muted);
        font-size: 0.95rem;
    }}
    .subtle-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
    }}
    .mini-stat {{
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.95rem 1rem;
    }}
    .mini-stat-label {{
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 0.3rem;
    }}
    .mini-stat-value {{
        color: var(--text);
        font-weight: 750;
        font-size: 1.15rem;
    }}
    </style>
    """

    st.markdown(css_text, unsafe_allow_html=True)


def render_stat_card(title, value):
    # Small helper for the repeated mini-stat markup used in the result area.
    return f"""
    <div class="mini-stat">
        <div class="mini-stat-label">{title}</div>
        <div class="mini-stat-value">{value}</div>
    </div>
    """


def build_prediction_input(
    departure_city,
    arrival_city,
    flight_type,
    agency,
    travel_date,
    travel_time,
):
    # Start from a zero-filled feature row, then turn on only the values selected in the form.
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

    # Rebuild the final DataFrame in the exact feature order expected by the saved model.
    input_df = pd.DataFrame(
        [[input_row[column] for column in MODEL_FEATURE_COLUMNS]],
        columns=MODEL_FEATURE_COLUMNS,
    )

    return input_df


def build_route_summary(flights_df):
    # Route-level aggregates give the prediction card some historical context beyond the model output.
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


# Load and derive the base dashboard data once before the layout is rendered.
try:
    flights_df = load_flights_data()

    route_summary_df = build_route_summary(flights_df)

    # Keep form options tied directly to the source dataset so dropdowns and summaries stay consistent.
    city_options = sorted(flights_df["from"].dropna().unique())

    flight_type_options = sorted(flights_df["flightType"].dropna().unique())

    agency_options = sorted(flights_df["agency"].dropna().unique())

    dataset_min_date = pd.to_datetime(flights_df["date"]).min().date()

    dataset_max_date = pd.to_datetime(flights_df["date"]).max().date()

    route_count = flights_df[["from", "to"]].drop_duplicates().shape[0]

    average_price = float(flights_df["price"].mean())

    agency_count = flights_df["agency"].nunique()

    top_route_prices_df = (
        route_summary_df.groupby(["from", "to"], as_index=False)
        .agg(avg_price=("avg_price", "mean"))
        .sort_values(by="avg_price", ascending=False)
    )

    top_route_prices_df["route"] = (
        top_route_prices_df["from"] + " -> " + top_route_prices_df["to"]
    )

    agency_summary_df = (
        flights_df.groupby("agency", as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_time=("time", "mean"),
            trip_count=("travelCode", "count"),
        )
        .sort_values(by="avg_price", ascending=False)
    )

    agency_summary_df["avg_price"] = agency_summary_df["avg_price"].round(2)

    agency_summary_df["avg_time"] = agency_summary_df["avg_time"].round(2)
except FileNotFoundError as error:
    st.error("The flight dashboard could not start because a required runtime file is missing.")
    st.code(str(error))
    st.stop()
except Exception as error:
    st.error("The flight dashboard could not finish loading the dataset and summary views.")
    with st.expander("Open startup details", expanded=False):
        st.write(str(error))
    st.stop()

# Apply the saved theme before any visible layout is drawn.
selected_theme = get_selected_theme()

with st.sidebar:
    # The sidebar keeps page controls and data-source details out of the main prediction area.
    st.header("Flight Studio")

    selected_theme = st.radio(
        "Theme",
        options=["System", "Light", "Dark"],
        index=["System", "Light", "Dark"].index(selected_theme),
    )

    st.session_state.flight_theme_mode_v2 = selected_theme

    st.divider()
    st.write(
        "Estimate flight fare from route, cabin, agency, travel date, and journey time."
    )
    st.info(
        "This dashboard uses the saved flight price model and the travel capstone dataset."
    )

    if use_flight_api():
        st.success("Prediction source: Flask API")

        api_model_info = load_api_model_info()

        if api_model_info.get("run_name"):
            st.caption(f"MLflow run: {api_model_info['run_name']}")
    else:
        st.caption("Prediction source: local joblib model")

apply_theme_css(selected_theme)

# Lead with the project context so the page reads like a full demo instead of only a form.
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Voyage Analytics</div>
        <div class="hero-title">Flight Fare Studio</div>
        <div class="hero-copy">
            This dashboard estimates flight ticket prices using route, cabin, agency, date, and travel-time inputs.
            It also summarizes the selected route using the historical flight dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="large")
metric_col_1.metric("Flight Records", f"{len(flights_df):,}")
metric_col_2.metric("Distinct Routes", f"{route_count:,}")
metric_col_3.metric("Average Fare", format_currency(average_price))
metric_col_4.metric("Agencies", f"{agency_count}")

st.write("")

form_col, result_col = st.columns([1.05, 0.95], gap="large")

with form_col:
    # Keep all prediction inputs together in one form so the model only runs on submit.
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Plan A Journey</div>
            <div class="section-copy">Enter the route and travel details, then submit the form to estimate the ticket price.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("flight_prediction_form_v3", border=False):
        city_col_1, city_col_2 = st.columns(2)

        with city_col_1:
            departure_city = st.selectbox("From", options=city_options, index=0)

        with city_col_2:
            arrival_options = [city for city in city_options if city != departure_city]
            arrival_city = st.selectbox("To", options=arrival_options, index=0)

        detail_col_1, detail_col_2 = st.columns(2)

        with detail_col_1:
            flight_type = st.selectbox("Cabin", options=flight_type_options, index=0)

        with detail_col_2:
            agency = st.selectbox("Agency", options=agency_options, index=0)

        travel_col_1, travel_col_2 = st.columns(2)

        with travel_col_1:
            travel_date = st.date_input(
                "Travel date",
                value=dataset_min_date,
                min_value=dataset_min_date,
                max_value=dataset_max_date,
            )

        with travel_col_2:
            travel_time = st.slider(
                "Travel time in hours",
                min_value=float(flights_df["time"].min()),
                max_value=float(flights_df["time"].max()),
                value=float(round(flights_df["time"].median(), 2)),
                step=0.01,
            )

        submitted = st.form_submit_button("Estimate Ticket Price", use_container_width=True)

selected_route_df = route_summary_df[
    (route_summary_df["from"] == departure_city)
    & (route_summary_df["to"] == arrival_city)
    & (route_summary_df["flightType"] == flight_type)
]

if submitted:
    try:
        # Build the model-ready input row from the selected route and trip details.
        prediction_input_df = build_prediction_input(
            departure_city=departure_city,
            arrival_city=arrival_city,
            flight_type=flight_type,
            agency=agency,
            travel_date=travel_date,
            travel_time=travel_time,
        )

        if use_flight_api():
            # In deployed setups, send the request through the Flask service.
            api_result = predict_with_api(
                departure_city=departure_city,
                arrival_city=arrival_city,
                flight_type=flight_type,
                agency=agency,
                travel_date=travel_date,
                travel_time=travel_time,
            )
            predicted_price = float(api_result["predicted_price"])
        else:
            # Local fallback keeps the page usable even without the API container.
            model = load_model()
            predicted_price = float(model.predict(prediction_input_df)[0])

        # Session state keeps the latest result visible during the next rerun.
        st.session_state.flight_predicted_price_v3 = predicted_price
        st.session_state.flight_input_preview_v3 = prediction_input_df
        st.session_state.pop("flight_prediction_error_v3", None)

    except ModuleNotFoundError as error:
        st.session_state.flight_prediction_error_v3 = (
            "Prediction could not start because the `xgboost` package is not installed.",
            str(error),
        )

    except requests.RequestException as error:
        st.session_state.flight_prediction_error_v3 = (
            "The app could not connect to the Flask API for prediction.",
            str(error),
        )

    except RuntimeError as error:
        st.session_state.flight_prediction_error_v3 = (
            "The Flask API rejected the prediction request.",
            str(error),
        )

    except FileNotFoundError as error:
        st.session_state.flight_prediction_error_v3 = (
            "The saved model file is not available in this runtime.",
            str(error),
        )

    except Exception as error:
        st.session_state.flight_prediction_error_v3 = (
            "The app could not generate a prediction from the saved model.",
            str(error),
        )

with result_col:
    # The result panel always shows either a prediction, an error, or the idle state.
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Fare Result</div>
            <div class="section-copy">Review the predicted fare together with historical details for the selected route.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_error = st.session_state.get("flight_prediction_error_v3")
    predicted_price = st.session_state.get("flight_predicted_price_v3")

    if prediction_error:
        error_title, error_detail = prediction_error
        st.error(error_title)
        st.code("pip install xgboost")
        with st.expander("Open error details", expanded=False):
            st.write(error_detail)
    elif predicted_price is not None:
        price_difference = predicted_price - average_price
        comparison_text = "above" if price_difference >= 0 else "below"
        st.markdown(
            f"""
            <div class="result-shell">
                <div class="result-kicker">Estimated Fare</div>
                <div class="result-price">{format_currency(predicted_price)}</div>
                <div class="result-note">
                    {format_currency(abs(price_difference))} {comparison_text} the overall dataset average.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-shell">
                <div class="result-kicker">Estimated Fare</div>
                <div class="result-price">Ready</div>
                <div class="result-note">Submit the journey details to generate a model-based ticket price.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not selected_route_df.empty:
        selected_route_row = selected_route_df.iloc[0]
        route_col_1, route_col_2 = st.columns(2)
        route_col_1.metric(
            "Historical Avg Price",
            format_currency(float(selected_route_row["avg_price"])),
        )
        route_col_2.metric("Average Time", f"{selected_route_row['avg_time']:.2f} hours")

        route_col_3, route_col_4 = st.columns(2)
        route_col_3.metric(
            "Average Distance",
            f"{selected_route_row['avg_distance']:.2f} km",
        )
        route_col_4.metric("Observed Trips", f"{int(selected_route_row['trip_count'])}")
    else:
        st.info("No historical summary exists for this exact route and cabin.")

    if "flight_input_preview_v3" in st.session_state:
        with st.expander("Prediction input details", expanded=False):
            st.dataframe(
                st.session_state.flight_input_preview_v3,
                width="stretch",
                hide_index=True,
            )

st.write("")

insights_tab, data_tab = st.tabs(["Market Insights", "Dataset"])

with insights_tab:
    # These summaries help explain how the dataset behaves beyond a single prediction request.
    insight_col_1, insight_col_2 = st.columns([1.12, 0.88], gap="large")

    with insight_col_1:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Route Pricing Landscape</div>
                <div class="section-copy">These routes have the highest average ticket prices in the historical dataset.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.bar_chart(
            top_route_prices_df.head(10).set_index("route")["avg_price"],
            height=360,
        )

    with insight_col_2:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Agency Summary</div>
                <div class="section-copy">Average price, mean journey time, and trip counts by agency.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            agency_summary_df,
            width="stretch",
            hide_index=True,
        )

with data_tab:
    # Leave a quick dataset preview available for demos and manual sanity checks.
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Source Dataset Preview</div>
            <div class="section-copy">Preview the source records used for route analysis and model input preparation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_rows = st.slider(
        "Rows to preview",
        min_value=5,
        max_value=30,
        value=12,
    )
    st.dataframe(
        flights_df.head(preview_rows),
        width="stretch",
        hide_index=True,
    )


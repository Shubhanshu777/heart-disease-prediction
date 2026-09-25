"""Streamlit UI for the SVM-based heart disease risk screening model.

Preprocessing mirrors the training notebook, which standardizes features in TWO
stages:

1. A StandardScaler is fit on the five raw numeric columns
   (Age, RestingBP, Cholesterol, MaxHR, Oldpeak). This scaler was NOT saved, so
   it is reconstructed here from ``heart.csv`` using the notebook's cleaning
   steps (see ``build_numeric_scaler``).
2. A second StandardScaler (``scaler.pkl``) was fit on the full 18-feature matrix
   AFTER stage 1 and is what the model consumes.

Skipping stage 1 feeds raw magnitudes (e.g. Age=40) into a scaler that expects
already-standardized values, which pushes every input far from the decision
boundary and makes the model predict "high risk" for everyone. Both stages are
therefore applied here, in order, before ``model.predict``.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Plotly powers the risk gauge. It is optional: if it is not installed the app
# degrades gracefully to a plain progress bar (see the prediction section).
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE: bool = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Reserved status colors (good / warning / critical). These are used ONLY to
# signal risk state and always travel with a text label + icon, so meaning is
# never carried by color alone.
STATUS_COLOR_LOW: str = "#2f9e44"  # green  – low risk
STATUS_COLOR_MODERATE: str = "#f08c00"  # amber  – borderline
STATUS_COLOR_HIGH: str = "#e03131"  # red    – high risk
NEUTRAL_INK: str = "#6b7280"  # muted gray text that reads on light & dark

# --- Page configuration -------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Screening",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Light custom styling -----------------------------------------------------
# A small amount of CSS to give the app a cleaner, card-like feel without
# fighting Streamlit's default theming.
st.markdown(
    """
    <style>
        .app-header {
            text-align: center;
            padding: 0.5rem 0 1.5rem 0;
        }
        .app-header h1 { margin-bottom: 0.25rem; }
        .app-header p { color: #6b7280; font-size: 1.02rem; margin-top: 0; }
        div[data-testid="stForm"] {
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
        }
        .disclaimer {
            font-size: 0.85rem;
            color: #6b7280;
            border-top: 1px solid #e5e7eb;
            margin-top: 2rem;
            padding-top: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Directory that holds the serialized model artifacts (same folder as this file).
ARTIFACT_DIRECTORY: Path = Path(__file__).resolve().parent

# The five raw numeric columns standardized in stage 1 (see module docstring).
NUMERIC_FEATURE_COLUMNS: list[str] = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]


@st.cache_resource(show_spinner="Loading model…")
def load_model_artifacts() -> tuple[object, object, list[str]]:
    """Load and cache the trained model, the fitted scaler, and column order.

    Caching with ``st.cache_resource`` means these files are read from disk once
    per server process instead of on every widget interaction.
    """
    trained_model = joblib.load(ARTIFACT_DIRECTORY / "SVM_HEART.pkl")
    fitted_scaler = joblib.load(ARTIFACT_DIRECTORY / "scaler.pkl")
    training_column_order = list(joblib.load(ARTIFACT_DIRECTORY / "columns.pkl"))
    return trained_model, fitted_scaler, training_column_order


@st.cache_resource(show_spinner="Preparing preprocessing…")
def build_numeric_scaler() -> StandardScaler:
    """Reconstruct the stage-1 numeric scaler from ``heart.csv``.

    The training notebook fit a StandardScaler on the five numeric columns but
    never saved it. We rebuild it here by replaying the notebook's cleaning:
    drop duplicate rows, then replace physiologically-impossible zeros in
    Cholesterol and RestingBP with the mean of their non-zero values. Fitting on
    the same cleaned data reproduces the original scaler's mean/scale exactly.
    """
    raw_dataframe = pd.read_csv(ARTIFACT_DIRECTORY / "heart.csv").drop_duplicates()

    cholesterol_nonzero_mean = raw_dataframe.loc[
        raw_dataframe["Cholesterol"] != 0, "Cholesterol"
    ].mean()
    raw_dataframe["Cholesterol"] = (
        raw_dataframe["Cholesterol"].replace(0, cholesterol_nonzero_mean).round(2)
    )

    resting_bp_nonzero_mean = raw_dataframe.loc[
        raw_dataframe["RestingBP"] != 0, "RestingBP"
    ].mean()
    raw_dataframe["RestingBP"] = (
        raw_dataframe["RestingBP"].replace(0, resting_bp_nonzero_mean).round(2)
    )

    numeric_scaler = StandardScaler()
    numeric_scaler.fit(raw_dataframe[NUMERIC_FEATURE_COLUMNS])
    return numeric_scaler


def color_for_risk_score(risk_score: float) -> str:
    """Map a 0–1 risk score to its reserved status color band."""
    if risk_score < 0.40:
        return STATUS_COLOR_LOW
    if risk_score < 0.60:
        return STATUS_COLOR_MODERATE
    return STATUS_COLOR_HIGH


def build_risk_gauge(risk_score: float) -> "go.Figure":
    """Build a Plotly gauge for the model's heart-disease risk score.

    A gauge fits a single headline value with polarity. The moving bar takes the
    status color of its band; the surrounding bands are drawn as recessive tints
    so the value—not the backdrop—is what the eye lands on.
    """
    percentage: float = risk_score * 100
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={"suffix": "%", "font": {"size": 40, "color": NEUTRAL_INK}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": NEUTRAL_INK,
                    "tickfont": {"color": NEUTRAL_INK},
                },
                # The bar is the actual reading; it wears the status color.
                "bar": {"color": color_for_risk_score(risk_score), "thickness": 0.7},
                "borderwidth": 0,
                # Recessive tinted bands give context without competing for attention.
                "steps": [
                    {"range": [0, 40], "color": "rgba(47,158,68,0.15)"},
                    {"range": [40, 60], "color": "rgba(240,140,0,0.15)"},
                    {"range": [60, 100], "color": "rgba(224,49,49,0.15)"},
                ],
            },
        )
    )
    figure.update_layout(
        height=260,
        margin={"t": 10, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",  # transparent so it adapts to the theme
    )
    return figure


heart_disease_model, feature_scaler, training_column_order = load_model_artifacts()
numeric_scaler = build_numeric_scaler()

# --- Sidebar: model information ----------------------------------------------
with st.sidebar:
    st.header("ℹ️ About the model")
    st.markdown(
        "This app screens for heart disease risk using a trained "
        "**Support Vector Machine** classifier."
    )
    st.metric("Model type", type(heart_disease_model).__name__)
    st.metric("Input features", len(training_column_order))
    with st.expander("Feature list (training order)"):
        # The exact one-hot layout the scaler and model expect.
        st.write(training_column_order)
    st.caption(
        "Inputs are one-hot encoded, standardized in two stages (numeric scaler "
        "then the full-feature scaler), and classified. Predictions are only as "
        "good as the training data."
    )

# --- Header -------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🫀 Heart Disease Risk Screening</h1>
        <p>Enter the patient details below to estimate the risk of heart disease.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Input form ---------------------------------------------------------------
# Grouping every input inside a form avoids a full script rerun on each widget
# change; the model only runs when the user submits.
with st.form("patient_details_form"):
    st.subheader("👤 Demographics")
    demographics_left, demographics_right = st.columns(2)
    with demographics_left:
        patient_age: int = st.number_input(
            "Age (years)", min_value=18, max_value=100, value=40, step=1
        )
    with demographics_right:
        patient_sex: str = st.selectbox("Sex", ["FEMALE", "MALE"])

    st.subheader("🩺 Vitals & Blood Work")
    vitals_left, vitals_right = st.columns(2)
    with vitals_left:
        resting_blood_pressure: int = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=80, max_value=200, value=120
        )
        serum_cholesterol: int = st.number_input(
            "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200
        )
    with vitals_right:
        maximum_heart_rate: int = st.slider(
            "Maximum heart rate achieved (BPM)", min_value=60, max_value=220, value=150
        )
        fasting_blood_sugar_over_120: int = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda flag: "Yes" if flag == 1 else "No",
        )

    st.subheader("📈 ECG & Exercise")
    exercise_left, exercise_right = st.columns(2)
    with exercise_left:
        chest_pain_type: str = st.selectbox(
            "Chest pain type",
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: atypical angina · NAP: non-anginal pain · "
            "TA: typical angina · ASY: asymptomatic",
        )
        resting_ecg_result: str = st.selectbox(
            "Resting ECG", ["Normal", "ST", "LVH"]
        )
        exercise_induced_angina: str = st.selectbox(
            "Exercise-induced angina", ["NO", "YES"]
        )
    with exercise_right:
        st_depression_oldpeak: float = st.slider(
            "Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1
        )
        st_slope: str = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    predict_button_clicked: bool = st.form_submit_button(
        "🔍 Predict Risk", use_container_width=True, type="primary"
    )

# --- Prediction ---------------------------------------------------------------
if predict_button_clicked:
    # Build the raw feature dictionary using the exact one-hot column names the
    # model was trained on (see columns.pkl). Any name not present in the trained
    # column order is dropped by the reindex below; any missing one is filled 0.
    raw_feature_values: dict[str, float] = {
        "Age": patient_age,
        "Sex_F": 1 if patient_sex == "FEMALE" else 0,
        "ChestPainType_ATA": 1 if chest_pain_type == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain_type == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain_type == "TA" else 0,
        "ChestPainType_ASY": 1 if chest_pain_type == "ASY" else 0,
        "RestingBP": resting_blood_pressure,
        "Cholesterol": serum_cholesterol,
        "FastingBS": fasting_blood_sugar_over_120,
        "RestingECG_Normal": 1 if resting_ecg_result == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg_result == "ST" else 0,
        "RestingECG_LVH": 1 if resting_ecg_result == "LVH" else 0,
        "MaxHR": maximum_heart_rate,
        "ExerciseAngina_N": 1 if exercise_induced_angina == "NO" else 0,
        "Oldpeak": st_depression_oldpeak,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
        "ST_Slope_Down": 1 if st_slope == "Down" else 0,
    }

    # Align columns to the training order (extra dropped, missing filled with 0).
    input_dataframe: pd.DataFrame = pd.DataFrame([raw_feature_values]).reindex(
        columns=training_column_order, fill_value=0
    )

    # Stage 1: standardize the five raw numeric columns with the reconstructed
    # numeric scaler. Stage 2: apply the saved full-feature scaler. Applying both
    # in this order is what the model was trained on (see module docstring).
    input_dataframe[NUMERIC_FEATURE_COLUMNS] = numeric_scaler.transform(
        input_dataframe[NUMERIC_FEATURE_COLUMNS]
    )
    scaled_input: pd.DataFrame = pd.DataFrame(
        feature_scaler.transform(input_dataframe), columns=training_column_order
    )

    predicted_class: int = int(heart_disease_model.predict(scaled_input)[0])

    # Derive a 0–1 risk score for the gauge. If the model exposes calibrated
    # probabilities, use them; otherwise squash the SVM's signed decision margin
    # through a logistic function. The latter is an indicative score, NOT a
    # calibrated probability (this SVM was trained without probability=True).
    risk_score: float | None = None
    score_is_calibrated: bool = False
    if hasattr(heart_disease_model, "predict_proba"):
        risk_score = float(heart_disease_model.predict_proba(scaled_input)[0][1])
        score_is_calibrated = True
    elif hasattr(heart_disease_model, "decision_function"):
        decision_margin = float(heart_disease_model.decision_function(scaled_input)[0])
        risk_score = float(1.0 / (1.0 + np.exp(-decision_margin)))

    st.divider()
    if predicted_class == 1:
        st.error("### ⚠️ HIGH risk of heart disease")
    else:
        st.success("### ✅ LOW risk of heart disease")

    if risk_score is not None:
        # Prefer the richer gauge; fall back to a progress bar without Plotly.
        if PLOTLY_AVAILABLE:
            st.plotly_chart(build_risk_gauge(risk_score), use_container_width=True)
        else:
            st.progress(risk_score)
        score_label: str = (
            "Estimated probability of heart disease"
            if score_is_calibrated
            else "Risk score (indicative, not a calibrated probability)"
        )
        st.metric(score_label, f"{risk_score:.0%}")

    with st.expander("Review the values you entered"):
        # Values are cast to strings so the single mixed-type "Value" column
        # serializes cleanly to Arrow (numbers + text in one column otherwise
        # triggers a pyarrow conversion warning).
        entered_values: dict[str, str] = {
            "Age": str(patient_age),
            "Sex": patient_sex,
            "Chest pain type": chest_pain_type,
            "Resting BP (mm Hg)": str(resting_blood_pressure),
            "Cholesterol (mg/dL)": str(serum_cholesterol),
            "Fasting BS > 120": "Yes" if fasting_blood_sugar_over_120 else "No",
            "Resting ECG": resting_ecg_result,
            "Max HR (BPM)": str(maximum_heart_rate),
            "Exercise angina": exercise_induced_angina,
            "Oldpeak": str(st_depression_oldpeak),
            "ST slope": st_slope,
        }
        st.table(
            pd.DataFrame(
                {"Value": list(entered_values.values())},
                index=list(entered_values.keys()),
            )
        )

# --- Disclaimer ---------------------------------------------------------------
st.markdown(
    '<p class="disclaimer">⚕️ This tool is a machine-learning demo and is '
    "<strong>not</strong> a medical device. It does not provide a diagnosis. "
    "Always consult a qualified healthcare professional.</p>",
    unsafe_allow_html=True,
)

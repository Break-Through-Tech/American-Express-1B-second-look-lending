import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# The bundle stores LoanPreprocessor as "the class in preprocessor.py",
# so data/ must be importable before the bundle is loaded.
sys.path.append(str(REPO_ROOT / "data"))
import preprocessor  # noqa: E402

MODEL_PATH = APP_DIR / "model.joblib"
PERSONAS_PATH = APP_DIR / "personas.json"

# Cost model: repaid loan earns $2,000, default loses $8,000 -> breakeven PD = 20%.
GAIN_ON_REPAID = 2_000
LOSS_ON_DEFAULT = 8_000
APPROVAL_THRESHOLD = GAIN_ON_REPAID / (GAIN_ON_REPAID + LOSS_ON_DEFAULT)

DATE_COLS = ["date_decision", "birth_259D", "prev_approval_date_min", "prev_approval_date_max"]
CATEGORY_COLS = ["education_927M", "maritalstatus_703M", "description_5085714M", "classificationofcontr_13M"]


# Load the preprocessor + model bundle once and reuse it across reruns.
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


# Load the example applicants once.
@st.cache_data
def load_personas():
    return json.loads(PERSONAS_PATH.read_text())


# Turn one persona dict into a one-row table with the dtypes training saw:
# numbers as floats, missing categories as NaN, thin-file flag as bool.
def to_row(values: dict) -> pd.DataFrame:
    row = pd.DataFrame([values])
    for col in row.columns:
        if col in DATE_COLS:
            continue
        elif col in CATEGORY_COLS:
            row[col] = row[col].astype(object).where(row[col].notna(), np.nan)
        elif col == "is_thin_file":
            row[col] = row[col].astype(bool)
        else:
            row[col] = pd.to_numeric(row[col], errors="coerce").astype(float)
    return row


# Show negatives as -$555, not $-555.
def money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


st.title("Second-Look Lending: Default Risk Demo")
st.caption("Example applicants are built from group averages of synthetic data, not real people.")

bundle = load_bundle()
prep, model = bundle["preprocessor"], bundle["model"]
personas = load_personas()

# --- Applicant ---------------------------------------------------------
st.subheader("Applicant")
name = st.selectbox("Example applicant", list(personas.keys()))
values = dict(personas[name])
is_thin_file = bool(values["is_thin_file"])
st.write("Credit history:", "**Thin-file** (no bureau record)" if is_thin_file else "**Established**")

# Keys include the persona name so the fields reset when the persona changes.
values["mainoccupationinc_384A"] = st.number_input(
    "Stated income ($)", min_value=0, max_value=500_000, step=1_000,
    value=int(values["mainoccupationinc_384A"]), key=f"income_{name}")
values["credamount_770A"] = st.number_input(
    "Requested credit amount ($)", min_value=0, max_value=500_000, step=1_000,
    value=int(values["credamount_770A"]), key=f"credit_{name}")
values["annuity_780A"] = st.number_input(
    "Periodic loan payment ($)", min_value=0, max_value=50_000, step=50,
    value=int(values["annuity_780A"]), key=f"annuity_{name}")

# --- Prediction --------------------------------------------------------
X = prep.transform(to_row(values))
predicted_pd = model.predict_proba(X)[0, 1]
expected_value = (1 - predicted_pd) * GAIN_ON_REPAID - predicted_pd * LOSS_ON_DEFAULT

st.subheader("Prediction")
st.metric("Predicted probability of default (PD)", f"{predicted_pd:.1%}")

# --- Decision ----------------------------------------------------------
st.sidebar.header("Policy settings")
established_threshold = st.sidebar.slider(
    "Established cutoff (approve if PD below)", 0.05, 0.40, APPROVAL_THRESHOLD, 0.01)
thin_file_threshold = st.sidebar.slider(
    "Thin-file cutoff (approve if PD below)", 0.05, 0.40, APPROVAL_THRESHOLD, 0.01)
threshold = thin_file_threshold if is_thin_file else established_threshold

st.subheader("Decision")
if predicted_pd < threshold:
    st.success(f"APPROVE (PD is under the {threshold:.0%} cutoff)")
else:
    st.error(f"DENY (PD is at or above the {threshold:.0%} cutoff)")
st.metric("Expected profit if approved", money(expected_value))

# Show every field as text: the column mixes numbers, dates and codes,
# and Streamlit's table format needs one type per column.
with st.expander("All applicant fields"):
    fields = pd.DataFrame({
        "field": list(values.keys()),
        "value": ["" if v is None else str(v) for v in values.values()],
    })
    st.dataframe(fields, hide_index=True)
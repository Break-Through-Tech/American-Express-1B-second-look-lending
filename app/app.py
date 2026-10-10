from pathlib import Path

import joblib
import streamlit as st
import pandas as pd

# The model file sits next to this script. Building the path from
# __file__ means it works locally and on Streamlit Cloud.
MODEL_PATH = Path(__file__).resolve().parent / "sample_model.joblib"


# Load the saved model bundle once and re-use it across re-runs.
@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


st.title("Second-Look Lending: Sample Model")

bundle = load_model_bundle()
model = bundle["model"]
features = bundle["features"]

st.subheader("Applicant details")

# Each widget returns its current value; Streamlit reruns the script
# whenever any widget changes.
income = st.number_input(
    "Stated annual income ($)", min_value=0, max_value=500_000, value=30_000, step=1_000
)
credit_amount = st.number_input(
    "Requested credit amount ($)", min_value=0, max_value=500_000, value=20_000, step=1_000
)
annuity = st.number_input(
    "Periodic loan payment ($)", min_value=0, max_value=50_000, value=800, step=50
)
days_employed = st.number_input(
    "Days employed", min_value=0, max_value=20_000, value=1_000, step=100
)
is_thin_file = st.checkbox("Thin-file applicant, meaning no credit bureau history")

st.write("You entered:")
st.write(
    {
        "income": income,
        "credit_amount": credit_amount,
        "annuity": annuity,
        "days_employed": days_employed,
        "is_thin_file": is_thin_file,
    }
)

# Build one row in the exact order the model was trained on. The dict
# keys must match the feature names saved inside the model bundle.
input_values = {
    "mainoccupationinc_384A": income,
    "credamount_770A": credit_amount,
    "annuity_780A": annuity,
    "days_employed_700P": days_employed,
    "is_thin_file": int(is_thin_file),  # True/False -> 1/0, same as training
}
input_row = pd.DataFrame([input_values])[features]

# predict_proba returns [P(repay), P(default)]; [0, 1] picks the PD.
predicted_pd = model.predict_proba(input_row)[0, 1]

st.subheader("Prediction")
st.metric("Predicted probability of default (PD)", f"{predicted_pd:.1%}")

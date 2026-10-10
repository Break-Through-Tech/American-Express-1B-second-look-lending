from pathlib import Path

import joblib
import streamlit as st

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

st.write("Model loaded. It expects these inputs:")
st.write(features)
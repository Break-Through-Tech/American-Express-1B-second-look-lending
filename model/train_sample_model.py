import sys
from pathlib import Path

# Have the ability to import ./data/load_data.py 
# from this script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "data"))
import load_data

# The exact inputs the model expects. The Streamlit app must supply
# these same columns in this order.
FEATURES = [
    "mainoccupationinc_384A", # Stated income.
    "credamount_770A", # Requested credit amount.
    "annuity_780A", # Periodic payment.
    "days_employed_700P", # Employment length.
    "is_thin_file", # No bureau history.
]

# Train on weeks before this one, validate on this week and later.
VALIDATION_START_WEEK = 80

# The trained model is saved inside the repo because the file 
# is tiny so Streamlit Cloud can find it next to app.py at deploy time.
MODEL_PATH = REPO_ROOT / "app" / "sample_model.joblib"

def main():
    train = load_data.load_or_build_dataset("train")
    print(train[FEATURES].dtypes)
    print(f"Weeks in train: {train['WEEK_NUM'].min()} to {train['WEEK_NUM'].max()}")

if __name__ == "__main__":
    main()

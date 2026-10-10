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

# Select the model's input columns from the full merged table.
# Used for training now and mirrored by the app later.
#   prepare_features(train)
def prepare_features(df):
    # Use copy() so we never modify the caller's table.
    X = df[FEATURES].copy()

    # Booleans become 0/1 so training and the app feed 
    # the same numeric type.
    X["is_thin_file"] = X["is_thin_file"].astype(int)

    return X

# Split the merged table by time and not randomly. 
# Earlier weeks to fit the model, later weeks to check it. 
#   split_by_time(train)
def split_by_time(df):
    fit_df = df[df["WEEK_NUM"] < VALIDATION_START_WEEK]
    val_df = df[df["WEEK_NUM"] >= VALIDATION_START_WEEK]

    return (fit_df, val_df)

def main():
    train = load_data.load_or_build_dataset("train")
    fit_df, val_df = split_by_time(train)

    print(f"Fit: {len(fit_df)} rows, weeks {fit_df['WEEK_NUM'].min()}-{fit_df['WEEK_NUM'].max()}")
    print(f"Val: {len(val_df)} rows, weeks {val_df['WEEK_NUM'].min()}-{val_df['WEEK_NUM'].max()}")

    print("\nDefault rate by thin-file (fit):")
    print(fit_df.groupby("is_thin_file")["target"].mean())
    print("\nDefault rate by thin-file (val):")
    print(val_df.groupby("is_thin_file")["target"].mean())

if __name__ == "__main__":
    main()

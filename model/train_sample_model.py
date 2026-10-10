import sys
import lightgbm as lgb
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

# Train a LightGBM classifier on the fit set. Returns the fitted model.
#   train_model(fit_df)
def train_model(fit_df):
    model = lgb.LGBMClassifier(
        n_estimators=200, # Number of small trees to build.
        learning_rate=0.05, # How much each tree is allowed to correct.
        num_leaves=31, # How detailed each tree can get.
        random_state=42, # Same result every run.
        verbose=-1, # Silence LightGBM's progress chatter.
    )
    model.fit(prepare_features(fit_df), fit_df["target"])

    return model

def main():
    train = load_data.load_or_build_dataset("train")
    fit_df, val_df = split_by_time(train)

    model = train_model(fit_df)
    print("Model trained.")

    # PD = probability of default. [:, 1] picks the "default" column.
    sample_pd = model.predict_proba(prepare_features(val_df.head(5)))[:, 1]
    print("PD for first 5 validation loans:", sample_pd.round(3))
    print("Actual outcomes:                ", val_df["target"].head(5).tolist())

if __name__ == "__main__":
    main()

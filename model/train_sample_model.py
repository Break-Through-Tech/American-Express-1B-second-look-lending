import sys
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

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

# Score the model on the validation weeks with AUC, overall and per
# segment. Returns the overall AUC.
#   evaluate_model(model, val_df)
def evaluate_model(model, val_df):
    val_pd = model.predict_proba(prepare_features(val_df))[:, 1]
    y = val_df["target"].to_numpy()

    overall = roc_auc_score(y, val_pd)
    print(f"Validation AUC (all):         {overall:.4f}")

    # The project requires per-segment reporting: an overall number can
    # hide a weak thin-file result.
    for flag, name in [(False, "established"), (True, "thin-file")]:
        mask = (val_df["is_thin_file"] == flag).to_numpy()
        seg_auc = roc_auc_score(y[mask], val_pd[mask])
        print(f"Validation AUC ({name:<11}): {seg_auc:.4f}  (n={mask.sum()})")

    return overall

# Save the model together with its feature list so the app reads the
# expected inputs from the file instead of guessing.
#   save_model(model)
def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

def main():
    train = load_data.load_or_build_dataset("train")
    fit_df, val_df = split_by_time(train)

    model = train_model(fit_df)
    save_model(model)

    # Reload from disk and compare predictions on 1,000 loans.
    saved = joblib.load(MODEL_PATH)
    X = prepare_features(val_df.head(1000))
    before = model.predict_proba(X)[:, 1]
    after = saved["model"].predict_proba(X)[:, 1]

    print("Identical predictions:", (before == after).all())
    print("Saved features:", saved["features"])
    print(f"File size: {MODEL_PATH.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    main()

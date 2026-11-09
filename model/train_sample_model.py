import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Make data/ importable (load_data, features, preprocessor).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "data"))
import load_data  # noqa: E402
import preprocessor  # noqa: E402

# Train on weeks before this, validate on this week and later.
VALIDATION_START_WEEK = 80

# Bundle the app will load in step 5. The live app still uses sample_model.joblib.
MODEL_PATH = REPO_ROOT / "app" / "model.joblib"


# Earlier weeks fit, later weeks validate.
def split_by_time(df):
    fit_df = df[df["WEEK_NUM"] < VALIDATION_START_WEEK]
    val_df = df[df["WEEK_NUM"] >= VALIDATION_START_WEEK]
    return fit_df, val_df


# Fit the preprocessor on fit rows, then the model on its output.
# LightGBM is a placeholder until the team picks the final model.
def fit_pipeline(fit_df, exclude_cols=preprocessor.EXCLUDE_COLS):
    prep = preprocessor.LoanPreprocessor(exclude_cols=exclude_cols).fit(fit_df)
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
    )
    model.fit(prep.transform(fit_df), fit_df["target"].to_numpy())
    return prep, model


# AUC overall and per segment, plus mean predicted PD vs actual default rate.
def evaluate(prep, model, val_df, label):
    pd_val = model.predict_proba(prep.transform(val_df))[:, 1]
    y = val_df["target"].to_numpy()
    thin = val_df["is_thin_file"].to_numpy()

    print(f"\n{label}: {len(prep.feature_cols_)} features")
    print(f"  AUC (all):   {roc_auc_score(y, pd_val):.4f}")
    for flag, name in [(False, "established"), (True, "thin-file")]:
        m = thin == flag
        print(
            f"  {name:<11}: AUC {roc_auc_score(y[m], pd_val[m]):.4f}  "
            f"mean PD {pd_val[m].mean():.3f}  actual {y[m].mean():.3f}  (n={m.sum()})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-month", action="store_true",
                        help="also train a model that keeps MONTH, for comparison")
    args = parser.parse_args()

    train = load_data.load_or_build_dataset("train")
    fit_df, val_df = split_by_time(train)

    if args.compare_month:
        with_month = [c for c in preprocessor.EXCLUDE_COLS if c != "MONTH"]
        evaluate(*fit_pipeline(fit_df, exclude_cols=with_month), val_df, "WITH MONTH")

    prep, model = fit_pipeline(fit_df)
    evaluate(prep, model, val_df, "FINAL (no MONTH)")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": prep, "model": model}, MODEL_PATH)
    print(f"\nSaved bundle to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
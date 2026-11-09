"""
make_personas.py

Builds a few example applicants for the Streamlit app from group medians
of the training data. No real rows are copied: each persona value is a
summary of at least MIN_GROUP_SIZE applicants.
Output: app/personas.json (review it before committing).
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "data"))
import load_data  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "app" / "personas.json"

# Every persona must summarize at least this many applicants.
MIN_GROUP_SIZE = 1_000

DATE_COLS = ["date_decision", "birth_259D", "prev_approval_date_min", "prev_approval_date_max"]
CATEGORY_COLS = ["education_927M", "maritalstatus_703M", "description_5085714M", "classificationofcontr_13M"]
SKIP_COLS = ["case_id", "target"]


# Turn a group of applicants into one typical applicant.
# Numbers -> median, dates -> median date, categories -> most common value.
#   summarize(thin_file_rows)
def summarize(group: pd.DataFrame) -> dict:
    persona = {}
    for col in group.columns:
        if col in SKIP_COLS:
            continue
        s = group[col]

        if col in DATE_COLS:
            d = pd.to_datetime(s, errors="coerce").median()
            persona[col] = None if pd.isna(d) else d.strftime("%Y-%m-%d")
        elif col in CATEGORY_COLS:
            # Mostly-missing categories (e.g. bureau codes for thin-file) stay missing.
            most_common = s.mode(dropna=True)
            persona[col] = None if s.isna().mean() > 0.5 or most_common.empty else most_common.iloc[0]
        elif col == "is_thin_file":
            persona[col] = bool(s.mode().iloc[0])
        else:
            v = s.median()
            persona[col] = None if pd.isna(v) else round(float(v), 4)
    return persona


def main():
    df = load_data.load_or_build_dataset("train")
    est = df[df["is_thin_file"] == False]
    thin = df[df["is_thin_file"] == True]

    dpd = est["bureau_dpd_max"]
    loan_to_income = thin["credamount_770A"] / thin["mainoccupationinc_384A"]

    groups = {
        "Established - on-time payer": est[dpd <= dpd.quantile(0.10)],
        "Established - late payments": est[dpd >= dpd.quantile(0.90)],
        "Thin-file - typical": thin,
        "Thin-file - large loan for income": thin[loan_to_income >= loan_to_income.quantile(0.75)],
    }

    personas = {}
    for name, group in groups.items():
        print(f"{name}: summarizing {len(group):,} applicants")
        if len(group) < MIN_GROUP_SIZE:
            raise ValueError(f"{name}: only {len(group)} applicants, need {MIN_GROUP_SIZE}")
        personas[name] = summarize(group)

    # allow_nan=False guarantees valid JSON (missing values are written as null).
    OUTPUT_PATH.write_text(json.dumps(personas, indent=2, allow_nan=False))
    print(f"\nSaved {len(personas)} personas to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
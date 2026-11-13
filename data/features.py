"""
features.py

Fixed (non-fitted) feature steps, moved from the team's notebooks:
derived ratios/flags, missing flags, and the 8 extra missing indicators.
Same formulas for training, validation, test and the Streamlit app.
"""

import numpy as np
import pandas as pd

# Chosen in Handle_Missing's missingness analysis; hardcoded here
# so the analysis doesn't have to rerun at training/serving time.
EXTRA_INDICATOR_FEATURES = [
    "riskassesment_940T",
    "max_overdue_to_income",
    "bureau_a2_overdue_sum",
    "bureau_exposure_ratio",
    "credit_vs_bureau_history",
    "tax_declared_income_ratio",
    "prev_days_since_last_approval",
    "prior_credit_to_current",
]


# Run every fixed step in the notebook's order.
#   build_features(merged_df)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_applicant_feat(df)
    df = add_missingness_feat(df)
    df = add_extra_missing_indicators(df)
    return df


# Families 4-7: demographics, affordability, coverage, credit behavior.
def add_applicant_feat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["date_decision", "birth_259D", "prev_approval_date_min", "prev_approval_date_max"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Family 7: demographic / derived
    df["age_years"] = (df["date_decision"] - df["birth_259D"]).dt.days / 365.25
    employment_years = df["days_employed_700P"] / 365.25
    df["employment_age_ratio"] = employment_years / df["age_years"].replace(0, np.nan)
    df["income_per_employed_year"] = df["mainoccupationinc_384A"] / (df["days_employed_700P"] / 365.0 + 1)
    df["coapplicant_income_ratio"] = np.where(
        (df["person_count"] > 1) & (df["person_income_min"] > 0) & (df["person_income_max"] > 0),
        df["person_income_min"] / df["person_income_max"].replace(0, np.nan),
        np.nan,
    )

    # Family 5: affordability and leverage (replace 0 with NaN to avoid division by zero)
    income = df["mainoccupationinc_384A"].replace(0, np.nan)
    credit = df["credamount_770A"].replace(0, np.nan)
    annuity = df["annuity_780A"].replace(0, np.nan)
    bureau_credit = df["bureau_credit_amount_mean"].replace(0, np.nan)

    df["credit_income_ratio"] = df["credamount_770A"] / income
    df["annuity_income_ratio"] = df["annuity_780A"] / income
    df["implied_term"] = df["credamount_770A"] / annuity
    df["tax_declared_income_ratio"] = df["tax_amount_mean"] / income
    df["bureau_exposure_ratio"] = df["bureau_credit_amount_sum"] / income
    df["credit_concentration"] = df["bureau_credit_amount_max"] / df["bureau_credit_amount_sum"].replace(0, np.nan)
    df["prior_credit_to_current"] = df["prev_credit_amount_mean"] / credit

    # Family 4: coverage and presence
    df["has_coapplicant"] = (df["person_count"] > 1).astype(int)
    df["has_prev"] = (df["prev_application_count"].fillna(0) > 0).astype(int)
    df["has_tax"] = (df["tax_record_count"].fillna(0) > 0).astype(int)
    df["history_source_count"] = (
        (df["is_thin_file"].astype(bool) == False).astype(int) + df["has_prev"] + df["has_tax"]
    )

    # Family 6: credit behavior
    df["bureau_overdue_credit_ratio"] = df["bureau_overdue_amount_mean"] / bureau_credit
    df["queries_per_contract"] = df["numberofqueries_146L"] / (df["bureau_contract_count"].fillna(0) + 1)
    df["max_overdue_to_income"] = df["bureau_overdue_amount_max"] / income
    df["months_per_contract"] = df["bureau_a2_record_count"] / df["bureau_contract_count"].replace(0, np.nan)
    df["avg_overdue_per_contract"] = df["bureau_overdue_amount_sum"] / df["bureau_contract_count"].replace(0, np.nan)
    df["prev_days_since_last_approval"] = (df["date_decision"] - df["prev_approval_date_max"]).dt.days
    df["prev_days_since_first_approval"] = (df["date_decision"] - df["prev_approval_date_min"]).dt.days

    combined_credit_history = df["bureau_credit_amount_sum"].fillna(0) + df["prev_credit_amount_sum"].fillna(0)
    df["current_history_credit_ratio"] = df["credamount_770A"] / combined_credit_history.replace(0, np.nan)
    df["credit_vs_bureau_history"] = df["credamount_770A"] / bureau_credit

    return df


# Family 9: missing flags.
def add_missingness_feat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["missing_prev_history"] = df["prev_application_count"].isna().astype(int)
    df["missing_tax_history"] = df["tax_record_count"].isna().astype(int)
    df["missing_income"] = df["mainoccupationinc_384A"].isna().astype(int)
    df["missing_annuity"] = df["annuity_780A"].isna().astype(int)
    df["missing_days_employed"] = df["days_employed_700P"].isna().astype(int)
    df["missing_overdue"] = df["bureau_overdue_amount_mean"].isna().astype(int)
    return df


# Handle_Missing's 8 extra indicators: 1 = value was missing, 0 = present.
def add_extra_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in EXTRA_INDICATOR_FEATURES:
        df["missing_" + col] = df[col].replace([np.inf, -np.inf], np.nan).isna().astype("int8")
    return df
"""
preprocessor.py

Fitted preprocessing, moved from the team's notebooks: winsorization,
one-hot encoding, feature selection, and missing-value imputation.
fit() learns numbers from training rows; transform() applies them to any rows.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from features import build_features

# Amount columns clipped at the train 1st/99th percentiles (notebook cells 27-31).
AMOUNT_FEATURES = [
    "mainoccupationinc_384A", "credamount_770A", "annuity_780A",
    "person_income_sum", "person_income_mean", "person_income_max",
    "tax_amount_sum", "tax_amount_mean", "tax_amount_max",
    "prev_credit_amount_sum", "prev_credit_amount_mean", "prev_credit_amount_max",
    "bureau_credit_amount_sum", "bureau_credit_amount_mean", "bureau_credit_amount_max",
    "bureau_overdue_amount_mean", "bureau_overdue_amount_max",
    "bureau_a2_overdue_sum", "bureau_a2_overdue_max", "bureau_a2_overdue_mean",
]

# Masked categories to one-hot encode. Marital status is excluded:
# it was dropped by the fair-lending screen in the CA's feature catalog.
CATEGORICAL_COLS = ["education_927M", "description_5085714M", "classificationofcontr_13M"]

# Never model features. MONTH is a time signal, excluded like WEEK_NUM.
# ("target" is listed so training rows can be passed as-is.)
EXCLUDE_COLS = ["case_id", "WEEK_NUM", "MONTH", "target"]

# Imputation rules (Handle_Missing cell 48).
ZERO_FILL_COLS = [
    "tax_record_count", "prev_application_count", "bureau_contract_count",
    "bureau_a2_contract_count", "bureau_a2_record_count", "bureau_a2_overdue_valid_count",
]
RATIO_COL = "coapplicant_income_ratio"
STD_FEATURES = [("tax_amount_std", "tax_record_count"), ("prev_credit_amount_std", "prev_application_count")]


# Train-fitted preprocessing. Saved inside the model bundle.
#   prep = LoanPreprocessor().fit(train_rows)
#   X = prep.transform(any_rows)
class LoanPreprocessor:
    def __init__(self, categorical_cols=CATEGORICAL_COLS, exclude_cols=EXCLUDE_COLS):
        self.categorical_cols = list(categorical_cols)
        self.exclude_cols = list(exclude_cols)

    # Learn every number from the training rows only.
    def fit(self, df: pd.DataFrame):
        X = build_features(df)

        self.winsor_lower_ = X[AMOUNT_FEATURES].quantile(0.01)
        self.winsor_upper_ = X[AMOUNT_FEATURES].quantile(0.99)
        X = self._clip(X)

        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder_.fit(X[self.categorical_cols])
        X = self._encode(X)

        # Numeric/bool columns that actually vary in train.
        self.feature_cols_ = [
            c for c in X.select_dtypes(include=["number", "bool"]).columns
            if c not in self.exclude_cols and X[c].nunique(dropna=True) > 1
        ]

        ready = self._select(X)
        self.zero_fill_cols_ = [c for c in ZERO_FILL_COLS if c in self.feature_cols_]

        # Co-applicant ratio median, among applicants who have a co-applicant.
        median = ready.loc[X["has_coapplicant"] == 1, RATIO_COL].median()
        self.ratio_median_ = median if np.isfinite(median) else 0

        # Std fallback median, measured after the "fewer than 2 records -> 0" rule.
        temp = ready.copy()
        self._apply_std_zero_rule(temp, X)
        self.std_fallback_ = {}
        for std_col, _ in STD_FEATURES:
            m = temp[std_col].median()
            self.std_fallback_[std_col] = m if np.isfinite(m) else 0

        # Median imputer for everything without a special rule.
        special = self.zero_fill_cols_ + [RATIO_COL] + [c for c, _ in STD_FEATURES]
        self.median_cols_ = [c for c in self.feature_cols_ if c not in special]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(ready[self.median_cols_])

        return self

    # Apply fixed features + the saved numbers. Returns a model-ready matrix.
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._encode(self._clip(build_features(df)))
        ready = self._select(X)

        ready[self.zero_fill_cols_] = ready[self.zero_fill_cols_].fillna(0)

        ready.loc[X["has_coapplicant"] == 0, RATIO_COL] = 0
        ready[RATIO_COL] = ready[RATIO_COL].fillna(self.ratio_median_)

        self._apply_std_zero_rule(ready, X)
        for std_col, fallback in self.std_fallback_.items():
            ready[std_col] = ready[std_col].fillna(fallback)

        ready[self.median_cols_] = self.imputer_.transform(ready[self.median_cols_])
        return ready

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # --- helpers ---

    def _clip(self, X):
        X = X.copy()
        for col in AMOUNT_FEATURES:
            X[col] = X[col].clip(lower=self.winsor_lower_[col], upper=self.winsor_upper_[col])
        return X

    def _encode(self, X):
        encoded = pd.DataFrame(
            self.encoder_.transform(X[self.categorical_cols]),
            columns=self.encoder_.get_feature_names_out(self.categorical_cols),
            index=X.index,
        )
        return pd.concat([X.drop(columns=self.categorical_cols), encoded], axis=1)

    # Keep the model's columns, in a fixed order; inf -> NaN.
    def _select(self, X):
        ready = X.reindex(columns=self.feature_cols_).copy()
        return ready.replace([np.inf, -np.inf], np.nan)

    # Std is undefined with fewer than 2 records: set it to 0.
    @staticmethod
    def _apply_std_zero_rule(ready, X):
        for std_col, count_col in STD_FEATURES:
            ready.loc[X[count_col].fillna(0) < 2, std_col] = 0
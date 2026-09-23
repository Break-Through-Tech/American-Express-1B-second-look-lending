import os
import time
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

# Load the .env file where the script's configurations
# are housed.
load_dotenv()

# Put constants here.
DATA_DIR = Path(os.environ["DATA_DIR"])
CSV_DIR = DATA_DIR / "csv_files"

def main():
    # Start the processing timer.
    process_start = time.perf_counter()

    train_dataset = build_base_dataset("train")
    # test_dataset = build_base_dataset("test")

    print(f"Train Dataset Shape: {train_dataset.shape}")
    # print(f"Test Dataset Shape: {test_dataset.shape}")

    process_end = time.perf_counter()

    elapsed_time = process_end - process_start

    print(f"Elapsed time: {elapsed_time} seconds")

# Load, valie and merge the base table with the depth 0 tables.
def build_base_dataset(split: str) -> pd.DataFrame:
    # Load, validate base table.
    base = load_base(split)
    validate_depth0_table(base, "base")

    # Load, validate, merge base with static0 table.
    static_0 = load_static_0(split)
    validate_depth0_table(static_0, "static_0")
    merged = join_static_0(base, static_0)

    # Load, validate, merge base with static cb 0 table.
    static_cb_0 = load_static_cb_0(split)
    validate_depth0_table(static_cb_0, "static_cb_0")
    merged = join_static_cb_0(merged, static_cb_0)

    # Load, aggregate, merge tax_registry_a_1 table.
    tax_registry_a_1 = load_tax_registry_a_1(split)
    tax_registry_agg = aggregate_tax_registry_a_1(tax_registry_a_1)
    merged = join_tax_registry_a_1(merged, tax_registry_agg)

    # Load, aggregate, merge person_1 table.
    person_1 = load_person_1(split)
    person_agg = aggregate_person_1(person_1)
    merged = join_person_1(merged, person_agg)

    return merged

# Returns the split directory which should 
# be either split = "train" or "test".
#   get_split_dir("train")
def get_split_dir(split: str) -> Path:
    return CSV_DIR / split

# Load the base table, one row per loan, for
# the given split.
#   load_base("test")
def load_base(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_base.csv"

    return pd.read_csv(path)

# Sum a group but return NaN, not 0, if every value in the
# group is NaN. Prevents "no records" from looking identical
# to "records summed to zero".
def sum_min_count(x: pd.Series):
    return x.sum(min_count=1)

# Load and concatenate the static_0 table, which are
# split into two files.
#   load_static_0("train")
def load_static_0(split: str) -> pd.DataFrame:
    split_dir = get_split_dir(split)

    part_0 = pd.read_csv(split_dir / f"{split}_static_0_0.csv")
    part_1 = pd.read_csv(split_dir / f"{split}_static_0_1.csv")

    # The ignore index parameter is needed because without it the
    # concantenated DF will keep each part's original row index, 
    # meaning possible duplicate index 0 for example. Using it
    # re-numbers the index starting from 0 onward.
    return pd.concat([part_0, part_1], ignore_index=True)

# Validate the depth 0 tables to make sure there are no duplicate IDs.
#   validate_depth0_table(train_base, "base")
def validate_depth0_table(df: pd.DataFrame, name: str) -> None:
    dupes = df["case_id"].duplicated().sum()

    if dupes > 0:
        raise ValueError(f"{name}: found {dupes} duplicate case_id values")

    print(f"{name}: {len(df)} rows, no duplicate case_id, all is good!")

# Join the static 0 table with the base table on case ID.
# This is present only for about 70% of applicants.
#   join_static_0(test_base, test_static_0)
def join_static_0(base: pd.DataFrame, static_0: pd.DataFrame) -> pd.DataFrame:
    # Use Left Outer Join in order to use all keys from the left dataframe
    # and missing matches from the right dataframe are filled with NaN.
    # The one to one validation checks that the join keys are unique on
    # both sides and notifies immediately if, for example, a duplicate 
    # case_id snuck in.
    merged = base.merge(static_0, on="case_id", how="left", validate="one_to_one")

    # This is a row count check after merging just in case 
    # duplicate keys slipped through the cracks.
    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )
    
    return merged

# Join the static cb 0 table with the base table on case ID.
# About 30% of the rows will get NaN because these are the thin-file
# people with no history.
#   join_static_cb_0(train_static_cb_0, train_static_cb_0)
def join_static_cb_0(base: pd.DataFrame, static_cb_0: pd.DataFrame) -> pd.DataFrame:
    # Use Left Outer Join in order to use all keys from the left dataframe
    # and missing matches from the right dataframe are filled with NaN.
    merged = base.merge(static_cb_0, on="case_id", how="left", validate="one_to_one")

    # This is a row count check after merging just in case 
    # duplicate keys slipped through the cracks.
    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )
    
    return merged

# Load the static cb 0 table.
#   load_static_cb_0("train")
def load_static_cb_0(split: str) -> pd.DataFrame:    
    path = get_split_dir(split) / f"{split}_static_cb_0.csv"

    return pd.read_csv(path)

# Load the tax_registry_a_1 table. 
# Depth 1, meaning many rows per case_id.
#   load_tax_registry_a_1("train")
def load_tax_registry_a_1(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_tax_registry_a_1.csv"

    return pd.read_csv(path)

# Aggregate tax_registry_a_1 to one row per case_id.
#   aggregate_tax_registry_a_1(train_tax_registry_a_1)
def aggregate_tax_registry_a_1(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("case_id")
        .agg(
            tax_record_count=("num_group1", "count"),
            tax_amount_sum=("amount_4527230A", sum_min_count),
            tax_amount_mean=("amount_4527230A", "mean"),
            tax_amount_max=("amount_4527230A", "max"),
        )
        .reset_index()
    )

# Join the aggregated tax_registry_a_1 onto the merged base table.
#   join_tax_registry_a_1(merged, tax_registry_agg)
def join_tax_registry_a_1(base: pd.DataFrame, tax_registry_agg: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(tax_registry_agg, on="case_id", how="left", validate="one_to_one")

    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )

    return merged

# Load the person_1 table. 
# Depth 1, meaning 1 to 2 rows per case_id with applicant + optional co-applicant.
#   load_person_1("train")
def load_person_1(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_person_1.csv"

    return pd.read_csv(path)

# Aggregate person_1 to one row per case_id.
#   aggregate_person_1(train_person_1)
def aggregate_person_1(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("case_id")
        .agg(
            person_count=("num_group1", "count"),
            person_income_sum=("mainoccupationinc_384A", sum_min_count),
            person_income_mean=("mainoccupationinc_384A", "mean"),
            person_income_max=("mainoccupationinc_384A", "max"),
            employment_mean=("empl_employedtotal_800L", "mean"),
            employment_max=("empl_employedtotal_800L", "max")
        )
        .reset_index()
    )

# Join the aggregated person_1 onto the merged base table.
#   join_person_1(merged, person_agg)
def join_person_1(base: pd.DataFrame, person_agg: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(person_agg, on="case_id", how="left", validate="one_to_one")

    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )

    return merged

if __name__ == "__main__":
    main()

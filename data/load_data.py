import os
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
    train_dataset = build_base_dataset("train")
    test_dataset = build_base_dataset("test")

    print(f"Train Dataset Shape: {train_dataset.shape}")
    print(f"Test Dataset Shape: {test_dataset.shape}")

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
    # case ID snuck in.
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


if __name__ == "__main__":
    main()

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
    train_base = load_base("train")
    train_static_0 = load_static_0("train")

    print(train_base)
    print(train_static_0)

    validate_depth0_table(train_base, "base")
    validate_depth0_table(train_static_0, "static_0")

    train_merged = join_static_0(train_base, train_static_0)

    print(train_merged.shape)


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
def join_static_0(base: pd.DataFrame, static_0: pd.DataFrame) -> pd.DataFrame:
    # Use Left Outer Join in order to use all keys from the left dataframe
    # and missing matches from the right dataframe are filled with NaN.
    merged = base.merge(static_0, on="case_id", how="left")

    # This is a row count check after merging just in case 
    # duplicate keys slipped through the cracks.
    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )
    
    return merged


if __name__ == "__main__":
    main()

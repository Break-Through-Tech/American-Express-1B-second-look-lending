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


# Returns the split directory which should 
# be either split = "train" or "test".
#   get_split_dir("train")
#   get_split_dir("test")
def get_split_dir(split: str) -> Path:
    return CSV_DIR / split

# Load the base table, one row per loan, for
# the given split.
#   load_base("train")
#   load_base("test")
def load_base(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_base.csv"

    return pd.read_csv(path)

# Load and concatenate the static_0 table, which are
# split into two files.
#   load_static_0("train")
#   load_static_0("test")
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
#   validate_depth0_table(train_static_0, "static_0")
def validate_depth0_table(df: pd.DataFrame, name: str) -> None:
    dupes = df["case_id"].duplicated().sum()

    if dupes > 0:
        raise ValueError(f"{name}: found {dupes} duplicate case_id values")

    print(f"{name}: {len(df)} rows, no duplicate case_id, all is good!")


if __name__ == "__main__":
    main()

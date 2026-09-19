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
    pass
    # DELETE THIS LATER, IT'S FOR TESTING THIS SCRIPT
    # train_base = load_base("train")
    # print(train_base.shape)       # Expect (1_000_000, 5)
    # print(train_base.columns.tolist())

# Returns the split directory which should 
# be either split = "train" or "test".
def get_split_dir(split: str) -> Path:
    return CSV_DIR / split

# Load the base table, one row per loan, for
# the given split.
def load_base(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_base.csv"
    return pd.read_csv(path)


if __name__ == "__main__":
    main()

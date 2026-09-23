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

# Directory for cached merged datasets.
# Cached files should be ignored by git.
CACHE_DIR = DATA_DIR/ "cache"

def main():
    # Start the processing timer.
    process_start = time.perf_counter()

    train_dataset = load_or_build_dataset("train")
    test_dataset = load_or_build_dataset("test")

    print(f"Train Dataset Shape: {train_dataset.shape}\n")
    print(f"Test Dataset Shape: {test_dataset.shape}\n")

    # Stopping hte processing timer.
    process_end = time.perf_counter()
    elapsed_time = process_end - process_start

    print(f"Elapsed time: {elapsed_time} seconds\n")

def get_cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}_master.parquet"

def load_or_build_dataset(split: str, force_rebuild: bool = False) -> pd.DataFrame:
    cache_path = get_cache_path(split)

    if cache_path.exists() and not force_rebuild:
        print(f"Loading {split} from cache: {cache_path}")

        return pd.read_parquet(cache_path)

    print(f"Building {split} from raw CSVs if no cache found or force_rebuild=True...")
    
    df = build_base_dataset(split)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Cached {split} to: {cache_path}")

    return df

# Load, validate and merge the base table with the depth 0 tables.
def build_base_dataset(split: str) -> pd.DataFrame:
    print(f"Building the base dataset for {split}...\n")

    # Load, validate base table.
    base = load_base(split)
    validate_depth0_table(base, "base")
    print("Loaded base table.\n")

    # Load, validate, merge base with static0 table.
    static_0 = load_static_0(split)
    validate_depth0_table(static_0, "static_0")
    merged = join_static_0(base, static_0)
    print("Merged static_0 into base table.\n")

    # Load, validate, merge base with static cb 0 table.
    static_cb_0 = load_static_cb_0(split)
    validate_depth0_table(static_cb_0, "static_cb_0")
    merged = join_static_cb_0(merged, static_cb_0)
    print("Merged static_cb_0 into base table.\n")

    # Load, aggregate, merge tax_registry_a_1 table.
    tax_registry_a_1 = load_tax_registry_a_1(split)
    tax_registry_agg = aggregate_tax_registry_a_1(tax_registry_a_1)
    merged = join_tax_registry_a_1(merged, tax_registry_agg)
    print("Merged tax_registry_a_1 into base table.\n")

    # Load, aggregate, merge person_1 table.
    person_1 = load_person_1(split)
    person_agg = aggregate_person_1(person_1)
    merged = join_person_1(merged, person_agg)
    print("Merged person_1 into base table.\n")

    # Load, aggregate, merge applprev_1 table.
    applprev_1 = load_applprev_1(split)
    applprev_agg = aggregate_applprev_1(applprev_1)
    merged = join_applprev_1(merged, applprev_agg)
    print("Merged applprev_1 into base table.\n")

    # Load, aggregate, merge credit_bureau_a_1 table.
    credit_bureau_a_1 = load_credit_bureau_a_1(split)
    bureau_agg = aggregate_credit_bureau_a_1(credit_bureau_a_1)
    merged = join_credit_bureau_a_1(merged, bureau_agg)
    print("Merged credit_bureau_a_1 into base table.\n")

    # Thin-file flag with zero rows in the raw (pre-aggregation) bureau table.
    merged["is_thin_file"] = build_thin_file_flag(base, credit_bureau_a_1)
    # Make sure the thin-file flag column actually exists and has the
    # expected boolean values before moving on.
    print(f"Thin-file flag: {merged["is_thin_file"]}.\n")

    # Load, aggregate (two parts with subrecord to contract to case), merge
    # credit_bureau_a_2 table.
    credit_bureau_a_2 = load_credit_bureau_a_2(split)
    contract_agg = aggregate_credit_bureau_a_2_to_contract(credit_bureau_a_2)
    bureau_a2_case_agg = aggregate_credit_bureau_a_2_to_case(contract_agg)
    merged = join_credit_bureau_a_2(merged, bureau_a2_case_agg)
    print("Merged credit_bureau_a2_contract_case_agg into base table.\n")

    print(f"Finished building the base dataset for {split}!\n")

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

# Load the applprev_1 table.
# Depth 1, meaning 0 to4 rows per case_id for the applicant's
# own previous applications to this lender.
#   load_applprev_1("train")
def load_applprev_1(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_applprev_1.csv"

    return pd.read_csv(path)

# Aggregate applprev_1 to one row per case_id.
#   aggregate_applprev_1(train_applprev_1)
def aggregate_applprev_1(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy so as to not mutate the original DataFrame.
    df = df.copy()
    # Conver the string dates to real datetimes so our min/max will
    # sort chronologically and not alphabetically.
    # Any value that cannot be parsed will become NaT, Not a Time,
    # instead of crashing.
    df["approvaldate_319D"] = pd.to_datetime(df["approvaldate_319D"], errors="coerce")

    return (
        df.groupby("case_id")
        .agg(
            prev_application_count=("num_group1", "count"),
            prev_credit_amount_sum=("credamount_590A", sum_min_count),
            prev_credit_amount_mean=("credamount_590A", "mean"),
            prev_credit_amount_max=("credamount_590A", "max"),
            prev_approval_date_min=("approvaldate_319D", "min"),
            prev_approval_date_max=("approvaldate_319D", "max"),
        )
        .reset_index()
    )

# Join the aggregated applprev_1 onto the merged base table.
#   join_applprev_1(merged, applprev_agg)
def join_applprev_1(base: pd.DataFrame, applprev_agg: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(applprev_agg, on="case_id", how="left", validate="one_to_one")

    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )

    return merged

# Load and concatenate the credit_bureau_a_1 table, split into two files.
# This is the most important signal table where the thin-file applicants
# have zero rows in.
#   load_credit_bureau_a_1("train")
def load_credit_bureau_a_1(split: str) -> pd.DataFrame:
    split_dir = get_split_dir(split)

    part_0 = pd.read_csv(split_dir / f"{split}_credit_bureau_a_1_0.csv")
    part_1 = pd.read_csv(split_dir / f"{split}_credit_bureau_a_1_1.csv")

    return pd.concat([part_0, part_1], ignore_index=True)

# Aggregate credit_bureau_a_1 to one row per case_id.
#   aggregate_credit_bureau_a_1(train_credit_bureau_a_1)
def aggregate_credit_bureau_a_1(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("case_id")
        .agg(
            bureau_contract_count=("num_group1", "count"),
            bureau_credit_amount_sum=("credamount_770A", sum_min_count),
            bureau_credit_amount_mean=("credamount_770A", "mean"),
            bureau_credit_amount_max=("credamount_770A", "max"),
            bureau_overdue_amount_mean=("overdueamountmax_950A", "mean"),
            bureau_overdue_amount_max=("overdueamountmax_950A", "max"),
            bureau_dpd_max=("pmts_dpdvalue_108P", "max"),
            bureau_dpd_mean=("pmts_dpdvalue_108P", "mean"),
        )
        .reset_index()
    )

# Helper to compute the thin-file flag.
# True if this case_id has zero rows in the raw pre-aggregation credit_bureau_a_1 table, 
# meaning no bureau history at all.
#   build_thin_file_flag(train_base, train_credit_bureau_a_1)
def build_thin_file_flag(base: pd.DataFrame, credit_bureau_a_1: pd.DataFrame) -> pd.Series:
    has_bureau = base["case_id"].isin(credit_bureau_a_1["case_id"])

    is_thin_file = has_bureau == False

    return is_thin_file

# Join the aggregated credit_bureau_a_1 onto the merged base table.
#   join_credit_bureau_a_1(merged, bureau_agg)
def join_credit_bureau_a_1(base: pd.DataFrame, bureau_agg: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(bureau_agg, on="case_id", how="left", validate="one_to_one")

    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )

    return merged

# Load the credit_bureau_a_2 table. 
# Depth 2, meaning case_id to num_group1 (contract) to 
# num_group2 (payment/subrecord within that contract) which is the largest table.
#   load_credit_bureau_a_2("train")
def load_credit_bureau_a_2(split: str) -> pd.DataFrame:
    path = get_split_dir(split) / f"{split}_credit_bureau_a_2.csv"

    return pd.read_csv(path)

# Part 1: Collapse credit_bureau_a_2 from subrecord level (case_id, num_group1, num_group2) to 
# contract level (case_id, num_group1). One row per past credit 
# contract by summarizing its payment history.
#   aggregate_credit_bureau_a_2_to_contract(train_credit_bureau_a_2)
def aggregate_credit_bureau_a_2_to_contract(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["case_id", "num_group1"])
        .agg(
            contract_subrecord_count=("num_group2", "count"),
            contract_overdue_sum=("pmts_overdue_1140A", sum_min_count),
            contract_overdue_max=("pmts_overdue_1140A", "max"),
            contract_overdue_valid_count=("pmts_overdue_1140A", "count"),
        )
        .reset_index()
    )

# Part 2: Collapse contract-level credit_bureau_a_2 summaries from (case_id, num_group1) 
# to one row per case_id.
#   aggregate_credit_bureau_a_2_to_case(contract_agg)
def aggregate_credit_bureau_a_2_to_case(contract_agg: pd.DataFrame) -> pd.DataFrame:
    case_agg = (
        contract_agg.groupby("case_id")
        .agg(
            bureau_a2_contract_count=("num_group1", "count"),
            bureau_a2_record_count=("contract_subrecord_count", "sum"),
            bureau_a2_overdue_sum=("contract_overdue_sum", sum_min_count),
            bureau_a2_overdue_max=("contract_overdue_max", "max"),
            bureau_a2_overdue_valid_count=("contract_overdue_valid_count", "sum"),
        )
        .reset_index()
    )

    # Weighted mean, computed after. aggregation. This is the total overdue divided by
    # total valid records and not the average of each contract's own mean.
    # This avoids overweighting contracts that happen to have fewer subrecords.
    case_agg["bureau_a2_overdue_mean"] = (
        case_agg["bureau_a2_overdue_sum"] / case_agg["bureau_a2_overdue_valid_count"]
    )

    return case_agg

# Join both aggregated credit_bureau_a_2 parts into the merged base table.
#   join_credit_bureau_a_2(merged, bureau_a2_case_agg)
def join_credit_bureau_a_2(base: pd.DataFrame, bureau_a2_case_agg: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(bureau_a2_case_agg, on="case_id", how="left", validate="one_to_one")

    if len(merged) != len(base):
        raise ValueError(
            f"Row count changed after join: base had {len(base)}, "
            f"merged has {len(merged)}"
        )

    return merged

if __name__ == "__main__":
    main()

import sys
from pathlib import Path

# Have the ability to import ./data/load_data.py 
# from this script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "data"))
import load_data

def main():
    # Test to confirm the import works and data loads.
    print(f"Repo root: {REPO_ROOT}")
    train = load_data.load_or_build_dataset("train")
    print(f"Train shape: {train.shape}")

if __name__ == "__main__":
    main()

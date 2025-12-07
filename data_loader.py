"""
data_loader.py
Responsible for downloading the Kaggle dataset with kagglehub,
extracting and exposing helper functions to load CSV/parquet files
into pandas DataFrames.

Usage:
    from data_loader import download_dataset, list_dataset_files, load_table
"""

import os
import zipfile
import glob
from pathlib import Path
from typing import Optional
import pandas as pd

# kagglehub import for automatic downloads (user requested)
try:
    import kagglehub
except Exception as e:
    kagglehub = None
    # In deployment, ensure kagglehub is installed and configured.

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_ref: str = "yogendras843/online-casino-dataset", force: bool = False) -> Path:
    """
    Download dataset from Kaggle via kagglehub.
    Returns the path to extracted files (data/raw/<dataset_name>).
    - dataset_ref: Kaggle dataset slug
    - force: re-download even if files exist
    """
    if kagglehub is None:
        raise ImportError("kagglehub is not installed. Please `pip install kagglehub`.")

    target_dir = DATA_DIR / dataset_ref.replace("/", "_")
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        print(f"Dataset already exists at {target_dir}")
        return target_dir

    print("Downloading dataset via kagglehub; this may take a while ...")
    # kagglehub.dataset_download returns a path to a ZIP or folder depending on implementation
    download_path = kagglehub.dataset_download(dataset_ref)
    print("kagglehub returned:", download_path)

    # If it's a zip, extract
    download_path = Path(download_path)
    if download_path.is_file() and download_path.suffix in [".zip"]:
        print("Extracting zip to", target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(target_dir)
    elif download_path.is_dir():
        print("Moving downloaded folder to data directory.")
        # Some kagglehub versions return a folder path
        target_dir.mkdir(parents=True, exist_ok=True)
        # copy files (non-destructive)
        for f in download_path.iterdir():
            dest = target_dir / f.name
            if not dest.exists():
                try:
                    if f.is_dir():
                        # simple copytree alternative
                        os.system(f'cp -r "{f}" "{dest}"')
                    else:
                        os.system(f'cp "{f}" "{dest}"')
                except Exception:
                    pass
    else:
        raise RuntimeError(f"Unexpected download path: {download_path}")

    print("Dataset prepared at:", target_dir)
    return target_dir


def list_dataset_files(data_path: Optional[Path] = None, pattern: str = "**/*") -> list:
    """
    List files in the dataset folder. Useful to inspect what was downloaded.
    """
    data_path = Path(data_path) if data_path else (DATA_DIR / "yogendras843_online-casino-dataset")
    if not data_path.exists():
        # fallback: list any folder under data/raw
        paths = [p for p in DATA_DIR.iterdir() if p.is_dir()]
        if paths:
            data_path = paths[0]
        else:
            return []
    files = [p for p in data_path.glob(pattern) if p.is_file()]
    return [str(p.relative_to(data_path)) for p in files]


def _auto_read_file(path: Path) -> pd.DataFrame:
    """
    Small helper to read CSV / parquet with sane defaults.
    """
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_table(filename: str, data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a named table from the dataset folder (by file name).
    Example: load_table('players.csv')
    """
    # discover dataset folder
    if data_path:
        base = Path(data_path)
    else:
        # try to find the only dir under data/raw or a named one
        candidates = [p for p in DATA_DIR.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError("No dataset found. Call download_dataset() first.")
        # pick the first candidate (if multiple, user can pass data_path)
        base = candidates[0]

    # try direct path or glob
    target = base / filename
    if target.exists():
        return _auto_read_file(target)

    # try to find file anywhere under base
    matches = list(base.glob(f"**/{filename}"))
    if matches:
        return _auto_read_file(matches[0])

    # if not exact name, attempt to match by suffix or partial match
    all_files = list(base.glob("**/*"))
    # find files whose name contains filename string
    candidates = [p for p in all_files if p.is_file() and filename.lower() in p.name.lower()]
    if candidates:
        return _auto_read_file(candidates[0])

    raise FileNotFoundError(f"File {filename} not found in {base}. Available files: {list_dataset_files(base)}")

"""Download, extract, and clean the UCI SMS Spam Collection."""

from __future__ import annotations

import csv
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
DATA_FILE = RAW_DATA_DIR / "SMSSpamCollection"
LOCAL_DATA_FILE = PROJECT_DIR / "SMSSpamCollection.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset() -> Path:
    """Return the raw dataset path, downloading the official ZIP if needed."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return DATA_FILE

    if LOCAL_DATA_FILE.exists():
        shutil.copyfile(LOCAL_DATA_FILE, DATA_FILE)
        return DATA_FILE

    archive_path = RAW_DATA_DIR / "sms+spam+collection.zip"
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "SpamShield-NLP-Student-Project/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI dataset. Check your internet connection "
            f"and retry. Source: {DATASET_URL}"
        ) from exc

    archive_path.write_bytes(archive_bytes)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(
                (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
                None,
            )
            if member is None:
                raise RuntimeError("The UCI ZIP does not contain SMSSpamCollection.")
            DATA_FILE.write_bytes(archive.read(member))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded UCI dataset is not a valid ZIP file.") from exc

    return DATA_FILE


def load_dataset() -> pd.DataFrame:
    """Load cleaned SMS messages with `label` and `message` columns."""
    data = pd.read_csv(
        download_dataset(),
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
        dtype=str,
    )
    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].str.strip().str.lower()
    data["message"] = data["message"].str.strip()
    data = data[data["label"].isin(["ham", "spam"]) & data["message"].ne("")]
    data = data.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)

    if data.empty or set(data["label"].unique()) != {"ham", "spam"}:
        raise ValueError("The dataset must contain non-empty ham and spam messages.")
    return data

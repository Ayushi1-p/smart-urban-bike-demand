"""
project_config.py
Project ki saari settings ek jagah.
"""
from pathlib import Path

# ---------------- Paths ----------------
# __file__ = is file ka location. .parent.parent = project ka main folder.
# Isse code kisi bhi computer par chalega, path hard-code nahi karna padega.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HOUR_CSV = RAW_DATA_DIR / "hour.csv"   # main dataset (har ghante ka data)
DAY_CSV = RAW_DATA_DIR / "day.csv"     # sirf reference ke liye

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# ---------------- Reproducibility ----------------
# Har baar same result aaye, isliye random seed fix kar diya.
RANDOM_SEED = 42

# ---------------- Dataset source ----------------
DATASET_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
DATASET_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATASET_DOI = "10.24432/C5W894"
DATASET_LICENSE = "CC BY 4.0"
EXPECTED_HOUR_ROWS = 17389  # UCI website par likha number; asli file se verify karenge

# ---------------- Columns ----------------
HOUR_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]

TARGET = "cnt"   # jo predict karna hai: ek ghante me kitni bikes rent hui

# LEAKAGE GUARD - ye columns model ko KABHI input nahi dene:
#   cnt                -> ye to answer hi hai
#   casual, registered -> cnt = casual + registered, to inhe dena = answer bata dena
LEAKAGE_COLUMNS = ["cnt", "casual", "registered"]
# instant = sirf row number, dteday = date text -> ye bhi feature nahi
NON_FEATURE_COLUMNS = LEAKAGE_COLUMNS + ["instant", "dteday"]

# Har column ki allowed range (UCI documentation se) - data check karne ke liye
VALID_RANGES = {
    "season": (1, 4), "yr": (0, 1), "mnth": (1, 12), "hr": (0, 23),
    "holiday": (0, 1), "weekday": (0, 6), "workingday": (0, 1),
    "weathersit": (1, 4),
    "temp": (0.0, 1.0), "atemp": (0.0, 1.0), "hum": (0.0, 1.0), "windspeed": (0.0, 1.0),
}

# UCI ne temperature waghaira ko 0-1 me badal diya hai. Asli units me wapas
# lane ke liye ye numbers chahiye (dashboard me kaam aayenge):
TEMP_MIN_C, TEMP_MAX_C = -8.0, 39.0
ATEMP_MIN_C, ATEMP_MAX_C = -16.0, 50.0
HUM_MAX = 100.0
WINDSPEED_MAX = 67.0

# ---------------- Time-based split ----------------
# Model ko purane data par train karke naye data par test karenge (future se past nahi).
#   train      : 2011-01-01 se 2012-06-30
#   validation : 2012-07-01 se 2012-09-30  (best model chunne ke liye)
#   test       : 2012-10-01 se 2012-12-31  (final result, sirf ek baar)
TRAIN_END = "2012-06-30"
VALIDATION_END = "2012-09-30"
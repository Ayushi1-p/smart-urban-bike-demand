"""
data_loader.py
Kaam: hour.csv ko padhna aur CHECK karna ki file asli UCI format me hai.
Agar file galat ho, to code yahin ruk jaayega - baad me galat results nahi aayenge.

Run karne ka tarika (project folder se):  python -m src.data_loader
"""
import pandas as pd

from config import project_config as cfg


class DataValidationError(Exception):
    """Ye error tab aata hai jab data UCI format se match na kare."""


def validate_hour_data(df):
    """
    Data ko check karta hai. Jitni problems milti hain unki list return karta hai.
    Khaali list [] matlab data bilkul theek hai.
    """
    problems = []

    # Check 1: saare 17 columns hone chahiye
    missing_cols = [c for c in cfg.HOUR_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in cfg.HOUR_COLUMNS]
    if missing_cols:
        problems.append(f"Missing columns: {missing_cols}")
        return problems  # columns hi nahi hain to aage check nahi kar sakte
    if extra_cols:
        problems.append(f"Unexpected extra columns: {extra_cols}")

    # Check 2: har value apni range me ho (jaise hour 0-23, season 1-4)
    for col, (low, high) in cfg.VALID_RANGES.items():
        bad = (~df[col].between(low, high)).sum()
        if bad > 0:
            problems.append(f"{col}: {bad} values outside [{low}, {high}]")

    # Check 3: counts negative nahi ho sakte
    for col in ["casual", "registered", "cnt"]:
        if (df[col] < 0).any():
            problems.append(f"{col}: negative counts found")

    # Check 4: UCI ke hisaab se cnt = casual + registered hona chahiye
    mismatch = (df["casual"] + df["registered"] != df["cnt"]).sum()
    if mismatch > 0:
        problems.append(f"cnt != casual + registered in {mismatch} rows")

    # Check 5: dates sahi format me hon
    dates = pd.to_datetime(df["dteday"], errors="coerce")
    if dates.isna().any():
        problems.append(f"dteday: {dates.isna().sum()} invalid dates")

    return problems


def load_hour_data(path=cfg.HOUR_CSV, strict=True):
    """
    hour.csv padhta hai, check karta hai, date ko sahi type me badalta hai
    aur rows ko time ke order me sort karta hai.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} nahi mili. Download the official dataset from "
            f"{cfg.DATASET_DOWNLOAD_URL} aur hour.csv ko data/raw/ me rakho."
        )

    problems = validate_hour_data(df)
    if problems and strict:
        raise DataValidationError("Dataset failed validation:\n - " + "\n - ".join(problems))

    # dteday text ("2011-01-01") ke roop me aata hai -> asli date type me badlo
    df["dteday"] = pd.to_datetime(df["dteday"])
    # date + hour = poora timestamp. Sirf sorting/splitting ke liye, model input nahi.
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def summarise_data(df):
    """Report ke liye data ki basic jaankari ek dictionary me deta hai."""
    cols = [c for c in cfg.HOUR_COLUMNS if c in df.columns]
    # 2 saal me kitne ghante hone chahiye vs file me kitni rows hain
    total_hours = int((df["datetime"].max() - df["datetime"].min()) / pd.Timedelta(hours=1)) + 1

    return {
        "rows": len(df),
        "original_columns": len(cols),
        "missing_values_total": int(df[cols].isna().sum().sum()),
        # same data wali do rows (instant ko chhod ke, kyunki wo har row me alag hai)
        "duplicate_rows": int(df[[c for c in cols if c != "instant"]].duplicated().sum()),
        "duplicate_timestamps": int(df["datetime"].duplicated().sum()),
        "date_min": str(df["dteday"].min().date()),
        "date_max": str(df["dteday"].max().date()),
        "hours_in_date_range": total_hours,
        "missing_hours": total_hours - len(df),
        "cnt_min": int(df["cnt"].min()),
        "cnt_max": int(df["cnt"].max()),
        "cnt_mean": round(float(df["cnt"].mean()), 2),
        "cnt_median": float(df["cnt"].median()),
    }


if __name__ == "__main__":
    data = load_hour_data()
    print("Data load aur validate ho gaya!\n")
    print(data.head())          # pehli 5 rows
    print()
    print(data.dtypes)          # har column ka data type
    print()
    for key, value in summarise_data(data).items():
        print(f"{key}: {value}")
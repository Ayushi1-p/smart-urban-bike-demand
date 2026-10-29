"""
eda.py  -  Exploratory Data Analysis (syllabus Unit II)
Kaam: data ko graphs aur numbers se samajhna, model banane se PEHLE.
Saare graphs outputs/figures/ me PNG ban ke save hote hain (report me lagenge).

Run karne ka tarika (project folder se):  python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")  # graphs ko window me nahi, seedhe file me save karo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import project_config as cfg
from src.data_loader import load_hour_data

sns.set_theme(style="whitegrid")

# Codes ko padhne layak naam (sirf graphs ke liye)
SEASON_NAMES = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEATHER_NAMES = {1: "1 Clear", 2: "2 Mist/Cloudy", 3: "3 Light rain/snow", 4: "4 Heavy rain/snow"}


def save(fig, name):
    """Graph ko outputs/figures/<name>.png me save karke band karta hai."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.name}")


def add_readable_columns(df):
    """Graphs ke liye real units wale columns jodta hai. Original columns nahi badalte."""
    df = df.copy()
    df["temp_c"] = df["temp"] * (cfg.TEMP_MAX_C - cfg.TEMP_MIN_C) + cfg.TEMP_MIN_C
    df["season_name"] = df["season"].map(SEASON_NAMES)
    df["weather_name"] = df["weathersit"].map(WEATHER_NAMES)
    df["day_type"] = df["workingday"].map({1: "Working day", 0: "Weekend/Holiday"})
    return df


# ---------------------------------------------------------------- 1. Univariate
def plot_target_distribution(df):
    """cnt ka histogram: kitne ghanton me kitni demand thi."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["cnt"], bins=50, ax=ax, color="steelblue")
    ax.axvline(df["cnt"].mean(), color="red", linestyle="--", label=f"Mean = {df['cnt'].mean():.0f}")
    ax.axvline(df["cnt"].median(), color="green", linestyle="--", label=f"Median = {df['cnt'].median():.0f}")
    ax.set(title="Distribution of hourly rentals (cnt)", xlabel="Bikes rented in one hour", ylabel="Number of hours")
    ax.legend()
    save(fig, "01_target_distribution")


# ---------------------------------------------------------------- 2. Time patterns
def plot_hourly_pattern(df):
    """Har ghante ki average demand, working day vs weekend alag-alag."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=df, x="hr", y="cnt", hue="day_type", estimator="mean", errorbar=None, marker="o", ax=ax)
    ax.set(title="Average rentals by hour of day", xlabel="Hour of day (0-23)", ylabel="Average bikes rented",
           xticks=range(0, 24))
    ax.legend(title="Day type")
    save(fig, "02_hourly_pattern_by_daytype")


def plot_season_month(df):
    """Season aur month ke hisaab se average demand."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = ["Winter", "Spring", "Summer", "Fall"]
    sns.barplot(data=df, x="season_name", y="cnt", order=order, errorbar=None, ax=axes[0], color="steelblue")
    axes[0].set(title="Average rentals by season", xlabel="Season", ylabel="Average bikes per hour")
    sns.barplot(data=df, x="mnth", y="cnt", errorbar=None, ax=axes[1], color="steelblue")
    axes[1].set(title="Average rentals by month", xlabel="Month (1 = Jan)", ylabel="Average bikes per hour")
    save(fig, "03_season_and_month")


def plot_trend_over_time(df):
    """Mahine-dar-mahine average demand: 2011 se 2012 me growth dikhata hai."""
    monthly = df.set_index("datetime")["cnt"].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.axvline(pd.Timestamp(cfg.TRAIN_END), color="orange", linestyle="--", label="End of training period")
    ax.axvline(pd.Timestamp(cfg.VALIDATION_END), color="red", linestyle="--", label="End of validation period")
    ax.set(title="Monthly average rentals over time", xlabel="Month", ylabel="Average bikes per hour")
    ax.legend()
    save(fig, "04_trend_over_time")


# ---------------------------------------------------------------- 3. Weather
def plot_weather(df):
    """Weather type ke hisaab se demand (boxplot) aur temperature vs demand."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = [WEATHER_NAMES[k] for k in sorted(df["weathersit"].unique())]
    sns.boxplot(data=df, x="weather_name", y="cnt", order=order, ax=axes[0], color="lightblue")
    axes[0].set(title="Rentals by weather situation", xlabel="Weather", ylabel="Bikes rented in one hour")
    axes[0].tick_params(axis="x", rotation=15)

    sample = df.sample(min(3000, len(df)), random_state=cfg.RANDOM_SEED)  # 3000 points = saaf graph
    sns.scatterplot(data=sample, x="temp_c", y="cnt", alpha=0.3, ax=axes[1], s=12)
    axes[1].set(title="Temperature vs rentals (3,000 random hours)", xlabel="Temperature (°C)",
                ylabel="Bikes rented in one hour")
    save(fig, "05_weather_and_temperature")


# ---------------------------------------------------------------- 4. Multivariate
def plot_correlation(df):
    """Numeric columns ka correlation heatmap. casual/registered leakage dikhane ke liye shaamil hain."""
    cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
            "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap (Pearson)")
    save(fig, "06_correlation_heatmap")
    return corr


def plot_hour_weekday_heatmap(df):
    """Hour x weekday: ek saath do features ka asar (multivariate)."""
    table = df.pivot_table(index="weekday", columns="hr", values="cnt", aggfunc="mean")
    table.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(table, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Average bikes rented"})
    ax.set(title="Average rentals by weekday and hour", xlabel="Hour of day", ylabel="Weekday")
    save(fig, "07_weekday_hour_heatmap")


# ---------------------------------------------------------------- 5. Outliers
def outlier_report(df):
    """IQR method se cnt ke outliers ginta hai aur boxplot banata hai."""
    q1, q3 = df["cnt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    n_out = int((df["cnt"] > upper).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(x=df["cnt"], ax=axes[0], color="lightblue")
    axes[0].set(title="Boxplot of hourly rentals", xlabel="Bikes rented in one hour")
    out_hours = df.loc[df["cnt"] > upper, "hr"].value_counts().sort_index()
    axes[1].bar(out_hours.index, out_hours.values, color="indianred")
    axes[1].set(title="Hour of day for the high (IQR outlier) values", xlabel="Hour of day",
                ylabel="Number of outlier hours", xticks=range(0, 24))
    save(fig, "08_outliers")
    return {"Q1": q1, "Q3": q3, "IQR": iqr, "upper_limit": upper, "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / len(df), 2)}


# ---------------------------------------------------------------- 6. Missing hours
def missing_hours_report(df):
    """Kaunse ghante file me nahi hain, aur kis din sabse zyada gayab hain."""
    full = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    missing = full.difference(pd.DatetimeIndex(df["datetime"]))
    per_day = pd.Series(missing.date).value_counts()
    return missing, per_day


# ---------------------------------------------------------------- Run everything
def run_eda():
    df = add_readable_columns(load_hour_data())
    print(f"Loaded {len(df)} rows\n\nCreating figures in {cfg.FIGURES_DIR}")

    plot_target_distribution(df)
    plot_hourly_pattern(df)
    plot_season_month(df)
    plot_trend_over_time(df)
    plot_weather(df)
    corr = plot_correlation(df)
    plot_hour_weekday_heatmap(df)
    outliers = outlier_report(df)
    missing, per_day = missing_hours_report(df)

    # --------- Numbers to interpret (copy this output into the report) ---------
    print("\n=== KEY NUMBERS ===")
    print(f"Skewness of cnt: {df['cnt'].skew():.2f}  (0 = symmetric, >1 = strongly right-skewed)")

    print("\nAverage rentals by hour (top 5 busiest hours):")
    print(df.groupby("hr")["cnt"].mean().round(1).sort_values(ascending=False).head(5).to_string())

    print("\nAverage rentals: working day vs weekend/holiday:")
    print(df.groupby("day_type")["cnt"].mean().round(1).to_string())

    print("\nAverage rentals by season:")
    print(df.groupby("season_name")["cnt"].mean().round(1).to_string())

    print("\nAverage rentals by year (0 = 2011, 1 = 2012):")
    print(df.groupby("yr")["cnt"].mean().round(1).to_string())

    print("\nWeather situation - number of hours (class balance):")
    print(df["weathersit"].value_counts().sort_index().to_string())

    print("\nCorrelation of each column with cnt:")
    print(corr["cnt"].drop("cnt").round(3).sort_values(ascending=False).to_string())
    print(f"\nCorrelation temp vs atemp: {corr.loc['temp', 'atemp']:.3f}")

    print("\nOutliers (IQR method on cnt):")
    for k, v in outliers.items():
        print(f"  {k}: {round(float(v), 2)}")

    print(f"\nMissing hours: {len(missing)}")
    print("Days with the most missing hours:")
    print(per_day.head(8).to_string())


if __name__ == "__main__":
    run_eda()
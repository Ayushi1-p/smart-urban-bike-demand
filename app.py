"""
app.py - Streamlit dashboard (the "frontend").

Run from the project folder:   streamlit run app.py

The app does NOT train models. It reads:
    data/raw/hour.csv            -> dataset pages and interactive charts
    outputs/metrics/*.csv        -> model comparison tables (created by the training scripts)
    outputs/figures/*.png        -> saved figures
    models/*.joblib + *.json     -> trained pipelines, used through src/predict.py
"""
import altair as alt
import pandas as pd
import streamlit as st

from config import project_config as cfg
from src import predict as P
from src.data_loader import load_hour_data, summarise_data

st.set_page_config(page_title="Bike Demand Prediction", page_icon="🚲", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SEASONS = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEATHER = {1: "1 - Clear / few clouds", 2: "2 - Mist / cloudy",
           3: "3 - Light rain / light snow", 4: "4 - Heavy rain / snow / fog"}


# ------------------------------------------------------------------ cached loaders
@st.cache_data
def get_data():
    return load_hour_data()


@st.cache_data
def read_csv(name):
    path = cfg.METRICS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def show_figure(name, caption=None):
    path = cfg.FIGURES_DIR / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure {name}.png not found. Run the training/EDA scripts first.")


def load_meta():
    if not P.models_available():
        return None, None
    _, reg_meta, _, clf_meta = P.load_models()
    return reg_meta, clf_meta


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("🚲 Smart Urban Bike Demand Prediction System")
    st.caption("M.Tech Data Science & Analytics · CSE570 · Python, Data Science and Machine Learning")

    st.markdown("""
**Problem.** Bike-sharing demand changes sharply with the hour, season, weather and type of day.
If an operator under-estimates demand, stations run empty. If it over-estimates, bikes and
redistribution staff sit idle.

**What this system does.** Using two years of real hourly data, it learns how demand depends on
time and weather, and estimates demand for conditions you enter:

| Task | Type | Output |
|---|---|---|
| A | Regression | Expected number of bikes rented in one hour |
| B | Classification | Demand level: **Low / Medium / High** |

**Possible uses:** planning how many bikes to keep in service, timing redistribution and
maintenance, and studying how weather and calendar affect demand.

> This is a **decision-support prototype trained on historical data** (Washington D.C., 2011–2012).
> It is not connected to live stations or live weather. Its estimates describe that system,
> not any other city today.
""")

    reg_meta, clf_meta = load_meta()
    if reg_meta:
        st.subheader("Selected models (verified on the held-out test period, Oct–Dec 2012)")
        c1, c2 = st.columns(2)
        c1.metric("Regression model", reg_meta["selected_model"])
        c2.metric("Test MAE (bikes per hour)", f"{reg_meta['test_metrics']['MAE']:.1f}")
        c3, c4 = st.columns(2)
        c3.metric("Classification model", clf_meta["selected_model"])
        c4.metric("Test macro F1", f"{clf_meta['test_metrics']['f1_macro']:.3f}")
        st.caption(f"Regression target: {reg_meta['target_transform']} · "
                   f"Test RMSE {reg_meta['test_metrics']['RMSE']:.1f}, R² {reg_meta['test_metrics']['R2']:.3f} · "
                   f"Classification test accuracy {clf_meta['test_metrics']['accuracy']:.3f}, "
                   f"ROC-AUC {clf_meta['test_metrics']['roc_auc_ovr_macro']:.3f}")
    else:
        st.warning("Trained models not found. Run `python -m src.regression` and "
                   "`python -m src.classification`, then reload this page.")


def page_dataset(df):
    st.title("Dataset")
    st.markdown(f"""
**Source:** UCI Machine Learning Repository – *Bike Sharing* (ID 275) · [{cfg.DATASET_URL}]({cfg.DATASET_URL})
**Original system:** Capital Bikeshare, Washington D.C., USA · **Licence:** {cfg.DATASET_LICENSE} ·
**DOI:** {cfg.DATASET_DOI}

**Citation:** Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository.
Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background
knowledge. *Progress in Artificial Intelligence, 2*(2–3), 113–127.
""")
    s = summarise_data(df)
    c = st.columns(5)
    c[0].metric("Rows (hours)", f"{s['rows']:,}")
    c[1].metric("Original columns", s["original_columns"])
    c[2].metric("Missing values", s["missing_values_total"])
    c[3].metric("Duplicate rows", s["duplicate_rows"])
    c[4].metric("Hours with no record", s["missing_hours"])
    st.caption(f"Period {s['date_min']} to {s['date_max']}. Hourly rentals: min {s['cnt_min']}, "
               f"median {s['cnt_median']:.0f}, mean {s['cnt_mean']}, max {s['cnt_max']}.")

    st.subheader("Data dictionary")
    st.dataframe(pd.DataFrame([
        ["season", "Season", "1 Winter, 2 Spring, 3 Summer, 4 Fall", "Feature"],
        ["yr", "Year", "0 = 2011, 1 = 2012", "Feature"],
        ["mnth", "Month", "1–12", "Feature"],
        ["hr", "Hour of day", "0–23", "Feature"],
        ["holiday", "Public holiday", "0 / 1", "Feature"],
        ["weekday", "Day of week", "0 = Sunday … 6 = Saturday", "Feature"],
        ["workingday", "Working day", "1 = not weekend and not holiday", "Feature"],
        ["weathersit", "Weather", "1 clear … 4 heavy rain/snow (4 merged into 3)", "Feature"],
        ["temp", "Temperature", "normalised; °C = temp × 47 − 8", "Feature"],
        ["atemp", "Feels-like temperature", "normalised", "Not used (0.988 correlated with temp)"],
        ["hum", "Humidity", "normalised; % = hum × 100", "Feature"],
        ["windspeed", "Wind speed", "normalised; value = windspeed × 67", "Feature"],
        ["casual", "Rentals by casual users", "count", "Excluded – leakage"],
        ["registered", "Rentals by registered users", "count", "Excluded – leakage"],
        ["cnt", "Total rentals in the hour", "count = casual + registered", "Regression target"],
        ["hr_workingday", "Hour × day type (engineered)", "e.g. W17, N17", "Feature"],
        ["demand_level", "Low / Medium / High (engineered)", "from cnt, training thresholds", "Classification target"],
    ], columns=["Column", "Meaning", "Values", "Role"]), hide_index=True, width="stretch")

    st.subheader("First rows of the raw data")
    st.dataframe(df[cfg.HOUR_COLUMNS].head(10), hide_index=True, width="stretch")


def page_eda(df):
    st.title("Exploratory Data Analysis")

    st.subheader("Interactive: average rentals by hour")
    c1, c2 = st.columns(2)
    seasons = c1.multiselect("Season", list(SEASONS.values()), default=list(SEASONS.values()))
    year = c2.radio("Year", ["Both", "2011", "2012"], horizontal=True)
    sub = df[df["season"].map(SEASONS).isin(seasons)]
    if year != "Both":
        sub = sub[sub["yr"] == (0 if year == "2011" else 1)]
    if sub.empty:
        st.info("No data for this selection.")
    else:
        profile = (sub.assign(day_type=sub["workingday"].map({1: "Working day", 0: "Weekend/holiday"}))
                   .pivot_table(index="hr", columns="day_type", values="cnt", aggfunc="mean"))
        st.line_chart(profile, x_label="Hour of day", y_label="Average bikes rented per hour")
        st.caption(f"Based on {len(sub):,} hours. Working days show commute peaks around 8:00 and 17:00–18:00.")

    st.subheader("Interactive: average rentals by a chosen variable")
    var = st.selectbox("Group by", ["Weather situation", "Season", "Month", "Weekday", "Year"])
    mapping = {
        "Weather situation": df["weathersit"].map(WEATHER),
        "Season": df["season"].map(lambda x: f"{x} {SEASONS[x]}"),
        "Month": df["mnth"].map(lambda m: f"{m:02d} {MONTHS[m - 1]}"),
        "Weekday": df["weekday"].map(lambda d: f"{d} {WEEKDAYS[d]}"),
        "Year": df["yr"].map({0: "2011", 1: "2012"}),
    }
    grouped = df.groupby(mapping[var])["cnt"].agg(["mean", "count"]).rename(
        columns={"mean": "Average bikes per hour", "count": "Hours"})
    st.bar_chart(grouped["Average bikes per hour"])
    st.dataframe(grouped.round(1), width="stretch")

    st.subheader("Saved EDA figures")
    for name, caption in [
        ("01_target_distribution", "Hourly rentals are right-skewed (skewness 1.28)."),
        ("02_hourly_pattern_by_daytype", "Two commute peaks on working days; one midday peak on weekends."),
        ("03_season_and_month", "Winter has the lowest demand, summer the highest."),
        ("04_trend_over_time", "Demand grew ~63% from 2011 to 2012, so a chronological split is used."),
        ("05_weather_and_temperature", "Rain/snow lowers demand; warmer hours have more rentals."),
        ("06_correlation_heatmap", "casual/registered are leakage; temp and atemp are 0.988 correlated."),
        ("07_weekday_hour_heatmap", "Weekday × hour pattern."),
        ("08_outliers", "IQR outliers are genuine rush hours and were kept."),
    ]:
        with st.expander(caption):
            show_figure(name)


def page_models():
    st.title("Model comparison")
    reg_meta, clf_meta = load_meta()
    tab_r, tab_c = st.tabs(["Regression (bike count)", "Classification (demand level)"])

    with tab_r:
        st.markdown("""
Models were tuned with 5-fold **TimeSeriesSplit** on the training period (Jan 2011–Jun 2012),
**selected on the validation period** (Jul–Sep 2012) by lowest RMSE, then refit on train + validation
and evaluated **once** on the test period (Oct–Dec 2012).

- **MAE**: average error in bikes · **RMSE**: like MAE but punishes large errors more ·
  **R²**: share of variation explained (1 = perfect, 0 = no better than the average, negative = worse).
""")
        if reg_meta:
            st.success(f"Selected: **{reg_meta['selected_model']}** on **{reg_meta['target_transform']}** "
                       f"· params {reg_meta['best_params']}")
        split = read_csv("split_summary.csv")
        if split is not None:
            st.markdown("**Chronological split**")
            st.dataframe(split, hide_index=True, width="stretch")
        for title, name in [("Validation results", "regression_validation_results.csv"),
                            ("Test results", "regression_test_results.csv")]:
            table = read_csv(name)
            if table is not None:
                st.markdown(f"**{title}**")
                sort_col = "val_RMSE" if "val_RMSE" in table else "test_RMSE"
                st.dataframe(table.sort_values(sort_col), hide_index=True, width="stretch")
        show_figure("09_regression_validation_comparison")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("11_regression_actual_vs_predicted")
        with c2:
            show_figure("10_polynomial_degree_curve", "Polynomial degree vs error (under/overfitting check)")
        show_figure("12_regression_residuals")
        show_figure("13_regression_test_two_weeks")
        show_figure("14_regression_top_coefficients")

    with tab_c:
        st.markdown("""
Demand levels use **tertile thresholds computed from the training period only**. Models were tuned with
TimeSeriesSplit (score = macro F1), selected on validation by **macro F1** (all three classes weighted
equally), and tested once.
""")
        if clf_meta:
            t = clf_meta["thresholds"]
            st.success(f"Selected: **{clf_meta['selected_model']}** · params {clf_meta['best_params']} · "
                       f"Low < {t['low_upper']:.0f} ≤ Medium < {t['medium_upper']:.0f} ≤ High (bikes/hour)")
        dist = read_csv("class_distribution.csv")
        if dist is not None:
            st.markdown("**Class distribution (hours)**")
            st.dataframe(dist, hide_index=True, width="stretch")
        for title, name in [("Validation results", "classification_validation_results.csv"),
                            ("Test results", "classification_test_results.csv")]:
            table = read_csv(name)
            if table is not None:
                st.markdown(f"**{title}**")
                st.dataframe(table.sort_values("f1_macro", ascending=False), hide_index=True,
                             width="stretch")
        show_figure("16_classification_validation_comparison")
        show_figure("17_confusion_matrices_test")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("18_roc_curves_test")
        with c2:
            show_figure("19_random_forest_feature_importance")
        show_figure("15_class_distribution")


def condition_inputs(prefix):
    """Input widgets shared by both prediction forms. Returns a dict in real units."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**When**")
        hr = st.slider("Hour of day", 0, 23, 17, key=f"{prefix}_hr", help="17 = 5 pm to 6 pm")
        mnth = st.selectbox("Month", range(1, 13), index=5, format_func=lambda m: MONTHS[m - 1],
                            key=f"{prefix}_mnth")
        weekday = st.selectbox("Day of week", range(7), index=3, format_func=lambda d: WEEKDAYS[d],
                               key=f"{prefix}_wd")
        holiday = int(st.checkbox("Public holiday", key=f"{prefix}_hol"))
    with c2:
        st.markdown("**Season and system level**")
        default_season = (mnth - 1) // 3 + 1   # UCI seasons change around the 21st of Mar/Jun/Sep/Dec
        season = st.selectbox("Season", [1, 2, 3, 4], index=default_season - 1,
                              format_func=lambda s: SEASONS[s], key=f"{prefix}_season",
                              help="Pre-filled from the month. Change it for dates after the 20th of Mar/Jun/Sep/Dec.")
        yr = st.radio("Demand level of year", [1, 0], format_func=lambda y: "2012 (busier)" if y else "2011",
                      key=f"{prefix}_yr",
                      help="The system grew ~63% from 2011 to 2012. The model only knows these two years.")
        wd_flag = P.derive_workingday(weekday, holiday)
        st.caption(f"Working day: **{'Yes' if wd_flag else 'No'}** (derived from day of week and holiday)")
    with c3:
        st.markdown("**Weather**")
        weathersit = st.selectbox("Weather", [1, 2, 3, 4], format_func=lambda w: WEATHER[w],
                                  key=f"{prefix}_weather",
                                  help="Category 4 occurred in only 3 training hours, so it is treated as 3.")
        temp_c = st.slider("Temperature (°C)", -8.0, 39.0, 25.0, 0.5, key=f"{prefix}_temp")
        hum = st.slider("Humidity (%)", 0, 100, 55, key=f"{prefix}_hum")
        wind = st.slider("Wind speed (UCI scale, 0–67)", 0.0, 67.0, 13.0, 0.5, key=f"{prefix}_wind")
    return {"hr": hr, "season": season, "mnth": mnth, "weekday": weekday, "holiday": holiday,
            "yr": yr, "weathersit": weathersit, "temp_c": temp_c, "humidity_pct": hum, "windspeed": wind}


def historical_average(df, inputs):
    wd = P.derive_workingday(inputs["weekday"], inputs["holiday"])
    same = df[(df["hr"] == inputs["hr"]) & (df["workingday"] == wd) & (df["yr"] == inputs["yr"])]
    return same["cnt"].mean(), len(same)


def page_predict(df):
    st.title("Predict demand")
    if not P.models_available():
        st.error("Trained models not found. Run `python -m src.regression` and "
                 "`python -m src.classification` first.")
        return
    _, reg_meta, _, clf_meta = P.load_models()
    st.markdown("Enter **hypothetical conditions**. The saved models (trained on 2011–2012 Washington D.C. "
                "data) estimate demand for one hour under those conditions.")

    tab_r, tab_c = st.tabs(["🔢 Bike count (regression)", "🏷️ Demand level (classification)"])

    with tab_r:
        inputs = condition_inputs("reg")
        if st.button("Predict bike count", type="primary", key="btn_reg"):
            try:
                count = P.predict_count(inputs)
            except P.InvalidInputError as e:
                st.error(f"Invalid input: {e}")
            else:
                band = P.level_from_count(count, clf_meta["thresholds"])
                avg, n = historical_average(df, inputs)
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted rentals", f"{count:.0f} bikes", help="For one hour")
                c2.metric("Demand band", band, help="Using the training-period thresholds")
                c3.metric("Historical average, same hour & day type", f"{avg:.0f} bikes" if n else "n/a",
                          help=f"Mean of {n} real hours in the dataset with the same hour, day type and year")
                st.info(f"**How to read this:** under these conditions the model expects about "
                        f"**{count:.0f} rentals between {inputs['hr']}:00 and {inputs['hr'] + 1}:00**. "
                        f"On the held-out test period this model's average error (MAE) was "
                        f"**{reg_meta['test_metrics']['MAE']:.0f} bikes/hour**, and errors were largest at "
                        f"commute peaks. This is a point estimate; no confidence interval is computed.")

                # What-if: same conditions for every hour of the day
                hours = []
                for h in range(24):
                    hours.append(P.predict_count({**inputs, "hr": h}))
                curve = pd.DataFrame({"Hour": range(24), "Predicted bikes": hours})
                st.markdown("**Same conditions across the whole day**")
                line = alt.Chart(curve).mark_line(point=True).encode(
                    x=alt.X("Hour:Q", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickCount=24),
                            title="Hour of day"),
                    y=alt.Y("Predicted bikes:Q", title="Predicted bikes per hour"),
                    tooltip=["Hour", alt.Tooltip("Predicted bikes:Q", format=".0f")])
                marker = alt.Chart(curve[curve["Hour"] == inputs["hr"]]).mark_point(
                    size=150, color="red", filled=True).encode(x="Hour:Q", y="Predicted bikes:Q")
                st.altair_chart(line + marker, width="stretch")
                st.caption("Red dot = the hour you selected.")

    with tab_c:
        inputs_c = condition_inputs("clf")
        t = clf_meta["thresholds"]
        st.caption(f"Classes: Low < {t['low_upper']:.0f} bikes/hour ≤ Medium < {t['medium_upper']:.0f} ≤ High "
                   f"(tertiles of the training period).")
        if st.button("Predict demand level", type="primary", key="btn_clf"):
            try:
                label, proba = P.predict_level(inputs_c)
            except P.InvalidInputError as e:
                st.error(f"Invalid input: {e}")
            else:
                colour = {"Low": "🟦", "Medium": "🟧", "High": "🟥"}[label]
                st.metric("Predicted demand level", f"{colour} {label}")
                probs = pd.DataFrame({"Class": list(proba), "Probability": list(proba.values())})
                bars = alt.Chart(probs).mark_bar().encode(
                    x=alt.X("Class:N", sort=["Low", "Medium", "High"], title="Demand level"),
                    y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                            title="Model probability"),
                    color=alt.Color("Class:N", sort=["Low", "Medium", "High"], legend=None,
                                    scale=alt.Scale(range=["#4c78a8", "#f58518", "#e45756"])),
                    tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")])
                st.altair_chart(bars, width="stretch")
                p = proba[label]
                p_text = ">99%" if p > 0.995 else f"{p:.0%}"
                st.info(f"**How to read this:** {clf_meta['selected_model']} assigns the highest probability to "
                        f"**{label}** ({p_text}). On the test period this model's macro F1 was "
                        f"{clf_meta['test_metrics']['f1_macro']:.2f}; Medium was the hardest class, often "
                        f"confused with its neighbours. Probabilities show the model's relative confidence "
                        f"and are not guaranteed to be calibrated.")


def page_limitations():
    st.title("Data source, limitations and future work")
    st.markdown(f"""
### Data source
UCI Machine Learning Repository, *Bike Sharing* dataset ({cfg.DATASET_URL}), licence {cfg.DATASET_LICENSE}.
Hourly rentals from Capital Bikeshare, Washington D.C., 2011–2012, with weather and holiday information.

### Limitations
- **Historical and single-city.** Patterns are from Washington D.C. in 2011–2012. They do not directly
  apply to other cities (for example Indian cities, with different climate, festivals and commute
  habits) or to the same system today.
- **Not live.** No real-time bike or weather feed is connected. Predictions are for conditions you enter.
- **Weather assumed known.** Models use the observed weather of the hour. Real use would rely on a
  forecast, which adds error that was not measured here.
- **Year feature.** The model only knows 2011 and 2012 demand levels, so it cannot extrapolate growth.
- **Fixed class thresholds.** Low/Medium/High come from the training period. As demand grows, more
  hours become "High".
- **Test period** is one season (Oct–Dec 2012) and includes the Hurricane Sandy shutdown (29–30 Oct 2012).
- **System-wide only.** No station-level predictions.

### What genuine live operation would need
- A live station-status feed (for example a GBFS feed published by the operator).
- A weather-forecast API for the hours being predicted.
- Regular retraining on recent local data, with thresholds reviewed using past data only.

### Future work
- Retrain the same pipeline on Indian city data if an official hourly dataset becomes available, adding
  local features (monsoon, regional festivals).
- Tree-based regressors (Random Forest / Gradient Boosting) and a wider regularisation grid.
- Prediction intervals, permutation feature importance, station-level forecasting.
""")


# ------------------------------------------------------------------ main
def main():
    st.sidebar.title("🚲 Bike Demand")
    page = st.sidebar.radio("Go to", ["Overview", "Dataset", "EDA", "Model comparison",
                                      "Predict demand", "Source & limitations"])
    st.sidebar.caption("Prototype trained on historical UCI data (Washington D.C., 2011–2012).")

    try:
        df = get_data()
    except (FileNotFoundError, Exception) as e:  # show a clear message instead of a crash
        st.error(f"Could not load data: {e}")
        return

    {"Overview": page_overview, "Dataset": lambda: page_dataset(df), "EDA": lambda: page_eda(df),
     "Model comparison": page_models, "Predict demand": lambda: page_predict(df),
     "Source & limitations": page_limitations}[page]()


if __name__ == "__main__":
    main()

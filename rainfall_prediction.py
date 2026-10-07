import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "combined_rainfall_weather.csv"

# Threshold for heavy/moderate rainfall prediction
RAIN_THRESHOLD = 20.0


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CONVERT DATE
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ============================================================
# 3. SORT DATA
# ============================================================

df = df.sort_values(
    ["District", "Date"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE / VERIFY TARGET
# ============================================================

# Tomorrow's rainfall should already exist
# from the dataset creation script.

if "next_day_rainfall_mm" not in df.columns:

    print(
        "\nnext_day_rainfall_mm not found."
    )

    print(
        "Creating it now..."
    )

    df["next_day_rainfall_mm"] = (
        df
        .groupby("District")[
            "imd_daily_rainfall_mm"
        ]
        .shift(-1)
    )


# ============================================================
# 5. CREATE RAIN / NO RAIN TARGET
# ============================================================

df["next_day_rain"] = (
    df["next_day_rainfall_mm"] > 0
).astype(int)


# ============================================================
# 6. CREATE >20 MM TARGET
# ============================================================

df["next_day_heavy_rain"] = (
    df["next_day_rainfall_mm"]
    > RAIN_THRESHOLD
).astype(int)


# ============================================================
# 7. SELECT FEATURES
# ============================================================

features = [

    "nasa_temperature_c",

    "nasa_humidity_percent",

    "nasa_surface_pressure_kpa",

    "nasa_wind_speed_ms",

    "nasa_precipitation_mm",

    "imd_daily_rainfall_mm",

    "previous_day_rainfall_mm",

    "rainfall_3day_avg_mm",

    "rainfall_7day_avg_mm",

    "month"

]


# Keep only columns that actually exist

features = [
    col
    for col in features
    if col in df.columns
]


print("\nFeatures used:")

for feature in features:
    print("  ", feature)


# ============================================================
# 8. PREPARE DATA
# ============================================================

model_df = df[
    features
    + [
        "next_day_rainfall_mm",
        "next_day_rain",
        "next_day_heavy_rain"
    ]
].copy()


# Replace infinite values

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove missing values

model_df = model_df.dropna()


print(
    "\nRows available for modelling:",
    len(model_df)
)


# ============================================================
# 9. INPUT AND TARGET
# ============================================================

X = model_df[features]

y_rain = model_df[
    "next_day_rain"
]

y_heavy = model_df[
    "next_day_heavy_rain"
]


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

# Use chronological split instead of random split.
# First 80% = training
# Last 20% = testing

split_index = int(
    len(model_df) * 0.80
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]


y_rain_train = y_rain.iloc[
    :split_index
]

y_rain_test = y_rain.iloc[
    split_index:
]


y_heavy_train = y_heavy.iloc[
    :split_index
]

y_heavy_test = y_heavy.iloc[
    split_index:
]


# ============================================================
# 11. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 12. MODEL 1 — PROBABILITY OF RAIN
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1: PROBABILITY OF RAIN")
print("=" * 70)


rain_model = LogisticRegression(
    max_iter=2000
)


rain_model.fit(
    X_train_scaled,
    y_rain_train
)


# Probability of rain

rain_probability = (
    rain_model
    .predict_proba(
        X_test_scaled
    )[:, 1]
)


rain_prediction = (
    rain_probability >= 0.5
).astype(int)


print(
    "\nAccuracy:",
    accuracy_score(
        y_rain_test,
        rain_prediction
    )
)


print(
    "\nROC-AUC:",
    roc_auc_score(
        y_rain_test,
        rain_probability
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_rain_test,
        rain_prediction
    )
)


# ============================================================
# 13. MODEL 2 — PROBABILITY OF >20 MM
# ============================================================

print("\n" + "=" * 70)
print(
    f"MODEL 2: PROBABILITY OF RAINFALL > "
    f"{RAIN_THRESHOLD} MM"
)
print("=" * 70)


heavy_model = LogisticRegression(
    max_iter=2000
)


heavy_model.fit(
    X_train_scaled,
    y_heavy_train
)


heavy_probability = (
    heavy_model
    .predict_proba(
        X_test_scaled
    )[:, 1]
)


heavy_prediction = (
    heavy_probability >= 0.5
).astype(int)


print(
    "\nAccuracy:",
    accuracy_score(
        y_heavy_test,
        heavy_prediction
    )
)


print(
    "\nROC-AUC:",
    roc_auc_score(
        y_heavy_test,
        heavy_probability
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_heavy_test,
        heavy_prediction
    )
)


# ============================================================
# 14. FIT WEIBULL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("WEIBULL DISTRIBUTION")
print("=" * 70)


# Use positive historical rainfall only

positive_rainfall = (
    df[
        df["imd_daily_rainfall_mm"] > 0
    ]["imd_daily_rainfall_mm"]
    .dropna()
    .values
)


# Fit Weibull with location fixed at zero

weibull_shape, weibull_loc, weibull_scale = (
    stats.weibull_min.fit(
        positive_rainfall,
        floc=0
    )
)


print(
    "\nShape (k):",
    weibull_shape
)

print(
    "Scale (lambda):",
    weibull_scale
)

print(
    "Location:",
    weibull_loc
)


# ============================================================
# 15. WEIBULL EXPECTED RAINFALL
# ============================================================

weibull_mean = stats.weibull_min.mean(
    weibull_shape,
    loc=0,
    scale=weibull_scale
)


weibull_median = stats.weibull_min.median(
    weibull_shape,
    loc=0,
    scale=weibull_scale
)


print(
    "\nExpected rainfall on a wet day:",
    round(weibull_mean, 2),
    "mm"
)


print(
    "Median rainfall on a wet day:",
    round(weibull_median, 2),
    "mm"
)


# ============================================================
# 16. HISTORICAL PROBABILITY OF RAIN
# ============================================================

total_days = len(
    df["imd_daily_rainfall_mm"]
    .dropna()
)


rainy_days = len(
    df[
        df["imd_daily_rainfall_mm"] > 0
    ]
)


historical_rain_probability = (
    rainy_days / total_days
)


print(
    "\nHistorical probability of rain:",
    round(
        historical_rain_probability * 100,
        2
    ),
    "%"
)


# ============================================================
# 17. FUNCTION TO CALCULATE WEIBULL PROBABILITY
# ============================================================

def weibull_probability_above(
    threshold
):

    return (
        1
        -
        stats.weibull_min.cdf(
            threshold,
            weibull_shape,
            loc=0,
            scale=weibull_scale
        )
    )


# ============================================================
# 18. DISPLAY HISTORICAL WEIBULL PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("WEIBULL RAINFALL PROBABILITIES")
print("=" * 70)


for threshold in [
    1,
    5,
    10,
    20,
    30,
    50,
    100
]:

    conditional_probability = (
        weibull_probability_above(
            threshold
        )
    )


    overall_probability = (
        historical_rain_probability
        *
        conditional_probability
    )


    print(

        f"\nP(Rain > {threshold} mm | wet day)"
        f" = "
        f"{conditional_probability * 100:.2f}%"

    )


    print(

        f"P(Rain > {threshold} mm)"
        f" = "
        f"{overall_probability * 100:.2f}%"

    )


# ============================================================
# 19. CREATE TEST RESULT TABLE
# ============================================================

results = X_test.copy()


results["Actual_Rain"] = (
    y_rain_test.values
)


results["Predicted_Rain_Probability"] = (
    rain_probability
)


results["Actual_Heavy_Rain"] = (
    y_heavy_test.values
)


results["Predicted_Heavy_Rain_Probability"] = (
    heavy_probability
)


results["Actual_Next_Day_Rainfall_mm"] = (
    model_df[
        "next_day_rainfall_mm"
    ]
    .iloc[
        split_index:
    ]
    .values
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

results.to_csv(
    "rainfall_prediction_results.csv",
    index=False
)


print(
    "\nPrediction results saved as:"
)

print(
    "rainfall_prediction_results.csv"
)


# ============================================================
# 21. FUNCTION FOR NEW WEATHER PREDICTION
# ============================================================

def predict_rainfall(
    temperature,
    humidity,
    pressure,
    wind_speed,
    nasa_precipitation,
    today_rainfall,
    previous_day_rainfall,
    rainfall_3day_avg,
    rainfall_7day_avg,
    month
):

    input_data = pd.DataFrame({

        "nasa_temperature_c":
            [temperature],

        "nasa_humidity_percent":
            [humidity],

        "nasa_surface_pressure_kpa":
            [pressure],

        "nasa_wind_speed_ms":
            [wind_speed],

        "nasa_precipitation_mm":
            [nasa_precipitation],

        "imd_daily_rainfall_mm":
            [today_rainfall],

        "previous_day_rainfall_mm":
            [previous_day_rainfall],

        "rainfall_3day_avg_mm":
            [rainfall_3day_avg],

        "rainfall_7day_avg_mm":
            [rainfall_7day_avg],

        "month":
            [month]

    })


    # Make sure columns have the same order

    input_data = input_data[
        features
    ]


    input_scaled = scaler.transform(
        input_data
    )


    # ----------------------------------------
    # Probability of rain
    # ----------------------------------------

    probability_rain = (
        rain_model
        .predict_proba(
            input_scaled
        )[0][1]
    )


    # ----------------------------------------
    # Probability of >20mm
    # ----------------------------------------

    probability_heavy = (
        heavy_model
        .predict_proba(
            input_scaled
        )[0][1]
    )


    # ----------------------------------------
    # Weibull expected rainfall
    # ----------------------------------------

    expected_rainfall_if_wet = (
        weibull_mean
    )


    # Expected rainfall for all days
    expected_rainfall = (
        probability_rain
        *
        expected_rainfall_if_wet
    )


    # ----------------------------------------
    # Probability > thresholds
    # ----------------------------------------

    probabilities = {}


    for threshold in [
        10,
        20,
        30,
        50,
        100
    ]:

        weibull_conditional = (
            weibull_probability_above(
                threshold
            )
        )


        # Combine weather-based probability
        # with Weibull conditional probability

        probability = (
            probability_rain
            *
            weibull_conditional
        )


        probabilities[
            threshold
        ] = probability


    return {

        "probability_of_rain":
            probability_rain,

        "probability_above_20mm_model":
            probability_heavy,

        "expected_rainfall_if_wet_mm":
            expected_rainfall_if_wet,

        "expected_rainfall_mm":
            expected_rainfall,

        "probability_above_10mm":
            probabilities[10],

        "probability_above_20mm":
            probabilities[20],

        "probability_above_30mm":
            probabilities[30],

        "probability_above_50mm":
            probabilities[50],

        "probability_above_100mm":
            probabilities[100]

    }


# ============================================================
# 22. EXAMPLE PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE WEATHER PREDICTION")
print("=" * 70)


example = predict_rainfall(

    temperature=28.0,

    humidity=82.0,

    pressure=998.0,

    wind_speed=3.5,

    nasa_precipitation=5.0,

    today_rainfall=10.0,

    previous_day_rainfall=8.0,

    rainfall_3day_avg=12.0,

    rainfall_7day_avg=10.0,

    month=7

)


print(
    "\nProbability of rain:"
)

print(
    f"{example['probability_of_rain'] * 100:.2f}%"
)


print(
    "\nProbability of rainfall >20mm:"
)

print(
    f"{example['probability_above_20mm_model'] * 100:.2f}%"
)


print(
    "\nExpected rainfall if it rains:"
)

print(
    f"{example['expected_rainfall_if_wet_mm']:.2f} mm"
)


print(
    "\nExpected rainfall overall:"
)

print(
    f"{example['expected_rainfall_mm']:.2f} mm"
)


print(
    "\nProbability >10mm:"
)

print(
    f"{example['probability_above_10mm'] * 100:.2f}%"
)


print(
    "\nProbability >20mm:"
)

print(
    f"{example['probability_above_20mm'] * 100:.2f}%"
)


print(
    "\nProbability >30mm:"
)

print(
    f"{example['probability_above_30mm'] * 100:.2f}%"
)


print(
    "\nProbability >50mm:"
)

print(
    f"{example['probability_above_50mm'] * 100:.2f}%"
)


print(
    "\nProbability >100mm:"
)

print(
    f"{example['probability_above_100mm'] * 100:.2f}%"
)


# ============================================================
# 23. PLOT PREDICTED RAIN PROBABILITY
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    rain_probability[:200],
    label="Predicted probability"
)

plt.plot(
    y_rain_test.values[:200],
    alpha=0.5,
    label="Actual rain"
)

plt.xlabel(
    "Test observation"
)

plt.ylabel(
    "Probability / actual rain"
)

plt.title(
    "Rainfall Prediction Results"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "rain_probability_prediction.png",
    dpi=150
)

plt.show()


print(
    "\nPlot saved as:"
)

print(
    "rain_probability_prediction.png"
)
# ============================================================
# WEATHER-BASED RAINFALL PREDICTION
# GLOBAL MODEL + WEIBULL DISTRIBUTION
# ============================================================

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "combined_rainfall_weather.csv"

RAIN_THRESHOLD = 20.0

OUTPUT_FILE = "global_prediction_results.csv"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

print(
    "Districts:",
    df["District"].nunique()
)

print(
    "States:",
    df["State"].nunique()
)


# ============================================================
# 2. DATE
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date"]
)


# ============================================================
# 3. CLEAN INVALID VALUES
# ============================================================

print("\nCleaning invalid values...")

# NASA POWER commonly uses -999 as missing/error value.
invalid_values = [
    -999,
    -999.0,
    -9999,
    -9999.0,
    np.inf,
    -np.inf
]

df = df.replace(
    invalid_values,
    np.nan
)

print(
    "Rows after removing invalid dates:",
    len(df)
)


# ============================================================
# 4. SORT CORRECTLY
# ============================================================

# VERY IMPORTANT:
# For next-day prediction, each district must be
# chronologically ordered first.

df = df.sort_values(
    ["District", "Date"]
).reset_index(drop=True)


# ============================================================
# 5. CREATE / RECREATE RAINFALL FEATURES
# ============================================================

rainfall_column = "imd_daily_rainfall_mm"

# Previous day's rainfall
df["previous_day_rainfall_mm"] = (
    df.groupby("District")[rainfall_column]
      .shift(1)
)

# 3-day rolling rainfall
df["rainfall_3day_avg_mm"] = (
    df.groupby("District")[rainfall_column]
      .transform(
          lambda x: x.rolling(
              window=3,
              min_periods=1
          ).mean()
      )
)

# 7-day rolling rainfall
df["rainfall_7day_avg_mm"] = (
    df.groupby("District")[rainfall_column]
      .transform(
          lambda x: x.rolling(
              window=7,
              min_periods=1
          ).mean()
      )
)


# ============================================================
# 6. CREATE NEXT-DAY TARGET
# ============================================================

df["next_day_rainfall_mm"] = (
    df.groupby("District")[rainfall_column]
      .shift(-1)
)


# ============================================================
# 7. RAIN / NO RAIN TARGET
# ============================================================

df["next_day_rain"] = np.where(
    df["next_day_rainfall_mm"].notna(),
    (
        df["next_day_rainfall_mm"] > 0
    ).astype(int),
    np.nan
)


# ============================================================
# 8. HEAVY RAIN TARGET
# ============================================================

df["next_day_heavy_rain"] = np.where(
    df["next_day_rainfall_mm"].notna(),
    (
        df["next_day_rainfall_mm"] > RAIN_THRESHOLD
    ).astype(int),
    np.nan
)


# ============================================================
# 9. CALENDAR FEATURES
# ============================================================

df["month"] = df["Date"].dt.month

df["day"] = df["Date"].dt.day

df["day_of_year"] = (
    df["Date"].dt.dayofyear
)


# ============================================================
# 10. FEATURES
# ============================================================

numeric_features = [

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


categorical_features = [

    "District",

    "State"

]


# Keep only features that actually exist
numeric_features = [
    x
    for x in numeric_features
    if x in df.columns
]

categorical_features = [
    x
    for x in categorical_features
    if x in df.columns
]


print("\nNumeric features:")

for feature in numeric_features:
    print("  ", feature)


print("\nCategorical features:")

for feature in categorical_features:
    print("  ", feature)


# ============================================================
# 11. CREATE MODEL DATASET
# ============================================================

required_columns = (

    ["Date"]

    +

    numeric_features

    +

    categorical_features

    +

    [
        "next_day_rainfall_mm",
        "next_day_rain",
        "next_day_heavy_rain"
    ]

)


model_df = df[
    required_columns
].copy()


# ============================================================
# 12. REMOVE MISSING VALUES
# ============================================================

# Remove invalid values again for safety
model_df = model_df.replace(
    [
        -999,
        -999.0,
        -9999,
        -9999.0,
        np.inf,
        -np.inf
    ],
    np.nan
)


# Drop rows where model features/targets are unavailable
model_df = model_df.dropna(
    subset=(
        numeric_features
        +
        categorical_features
        +
        [
            "next_day_rainfall_mm",
            "next_day_rain",
            "next_day_heavy_rain"
        ]
    )
)


print(
    "\nRows available for modelling:",
    len(model_df)
)


# ============================================================
# 13. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)


unique_dates = sorted(
    model_df["Date"].unique()
)


if len(unique_dates) < 10:

    raise ValueError(
        "Not enough dates available for training."
    )


split_position = int(
    len(unique_dates) * 0.80
)


train_end_date = unique_dates[
    split_position - 1
]

test_start_date = unique_dates[
    split_position
]


train_df = model_df[
    model_df["Date"] <= train_end_date
].copy()


test_df = model_df[
    model_df["Date"] >= test_start_date
].copy()


print(
    "\nTraining dates:",
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)


print(
    "Testing dates:",
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)


print(
    "\nTraining rows:",
    len(train_df)
)


print(
    "Testing rows:",
    len(test_df)
)


# ============================================================
# 14. X AND Y
# ============================================================

X_train = train_df[
    numeric_features +
    categorical_features
]


X_test = test_df[
    numeric_features +
    categorical_features
]


y_train_rain = train_df[
    "next_day_rain"
].astype(int)


y_test_rain = test_df[
    "next_day_rain"
].astype(int)


y_train_heavy = train_df[
    "next_day_heavy_rain"
].astype(int)


y_test_heavy = test_df[
    "next_day_heavy_rain"
].astype(int)


# ============================================================
# 15. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features
        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ]

)


# ============================================================
# 16. GLOBAL RAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL MODEL — PROBABILITY OF RAIN")
print("=" * 70)


rain_model = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=3000
            )
        )

    ]

)


rain_model.fit(
    X_train,
    y_train_rain
)


# ============================================================
# 17. TEST RAIN MODEL
# ============================================================

rain_probability = (
    rain_model
    .predict_proba(X_test)[:, 1]
)


rain_prediction = (
    rain_probability >= 0.5
).astype(int)


print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test_rain,
            rain_prediction
        ),
        4
    )
)


# ROC-AUC only if both classes exist
if len(
    np.unique(y_test_rain)
) > 1:

    print(
        "\nROC-AUC:",
        round(
            roc_auc_score(
                y_test_rain,
                rain_probability
            ),
            4
        )
    )


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test_rain,
        rain_prediction,
        zero_division=0
    )
)


# ============================================================
# 18. GLOBAL HEAVY RAIN MODEL
# ============================================================

print("\n" + "=" * 70)

print(
    f"GLOBAL MODEL — PROBABILITY OF > "
    f"{RAIN_THRESHOLD} MM"
)

print("=" * 70)


heavy_preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features
        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ]

)


heavy_model = Pipeline(

    steps=[

        (
            "preprocessor",

            heavy_preprocessor
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=3000
            )
        )

    ]

)


heavy_model.fit(
    X_train,
    y_train_heavy
)


# ============================================================
# 19. TEST HEAVY RAIN MODEL
# ============================================================

heavy_probability = (
    heavy_model
    .predict_proba(X_test)[:, 1]
)


heavy_prediction = (
    heavy_probability >= 0.5
).astype(int)


print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test_heavy,
            heavy_prediction
        ),
        4
    )
)


if len(
    np.unique(y_test_heavy)
) > 1:

    print(
        "\nROC-AUC:",
        round(
            roc_auc_score(
                y_test_heavy,
                heavy_probability
            ),
            4
        )
    )


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test_heavy,
        heavy_prediction,
        zero_division=0
    )
)


# ============================================================
# 20. GLOBAL WEIBULL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL WEIBULL DISTRIBUTION")
print("=" * 70)


positive_rainfall = (

    df[
        df[rainfall_column] > 0
    ][rainfall_column]

    .dropna()

    .values

)


if len(positive_rainfall) == 0:

    raise ValueError(
        "No positive rainfall observations found."
    )


# Fit Weibull
weibull_shape, weibull_loc, weibull_scale = (
    stats.weibull_min.fit(
        positive_rainfall,
        floc=0
    )
)


print(
    "\nWeibull shape (k):",
    round(
        weibull_shape,
        6
    )
)


print(
    "Weibull scale (lambda):",
    round(
        weibull_scale,
        6
    )
)


# ============================================================
# 21. WEIBULL MEAN
# ============================================================

weibull_mean = (
    stats.weibull_min.mean(
        weibull_shape,
        loc=0,
        scale=weibull_scale
    )
)


print(
    "\nExpected rainfall on wet day:",
    round(
        weibull_mean,
        2
    ),
    "mm"
)


# ============================================================
# 22. WEIBULL SURVIVAL FUNCTION
# ============================================================

def weibull_probability_above(
    threshold
):

    return stats.weibull_min.sf(
        threshold,
        weibull_shape,
        loc=0,
        scale=weibull_scale
    )


# ============================================================
# 23. DISPLAY DISTRIBUTION PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL WEIBULL PROBABILITIES")
print("=" * 70)


for threshold in [
    10,
    20,
    30,
    50,
    100
]:

    probability = (
        weibull_probability_above(
            threshold
        )
    )

    print(
        f"P(X > {threshold} mm | wet day)"
        f" = "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 24. CREATE TEST RESULTS
# ============================================================

results = test_df[
    [
        "Date",
        "State",
        "District",

        "nasa_temperature_c",

        "nasa_humidity_percent",

        "nasa_surface_pressure_kpa",

        "nasa_wind_speed_ms",

        "imd_daily_rainfall_mm",

        "next_day_rainfall_mm"
    ]
].copy()


results[
    "predicted_probability_of_rain"
] = rain_probability


results[
    "predicted_probability_above_20mm"
] = heavy_probability


results[
    "predicted_rain"
] = rain_prediction


# ============================================================
# 25. SAVE TEST RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nPrediction results saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 26. DISTRICT PREDICTION FUNCTION
# ============================================================

def predict_for_district(
    district_name
):

    # -----------------------------------------
    # Find district
    # -----------------------------------------

    district_data = df[
        df["District"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        district_name.strip().lower()
    ].copy()


    if len(district_data) == 0:

        print(
            "\nDistrict not found."
        )

        print(
            "\nExample districts:"
        )

        print(
            df["District"]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )

        return None


    # -----------------------------------------
    # Sort by date
    # -----------------------------------------

    district_data = (
        district_data
        .sort_values("Date")
        .copy()
    )


    # -----------------------------------------
    # Find latest VALID weather record
    # -----------------------------------------

    prediction_features = (
        numeric_features
        +
        categorical_features
    )


    valid_rows = district_data.copy()


    # Remove missing values
    valid_rows = valid_rows.dropna(
        subset=prediction_features
    )


    # Remove invalid placeholder values
    valid_rows = valid_rows[
        ~(
            valid_rows[
                numeric_features
            ]
            .isin(
                [
                    -999,
                    -999.0,
                    -9999,
                    -9999.0
                ]
            )
            .any(axis=1)
        )
    ]


    if valid_rows.empty:

        print(
            "\nNo valid weather data available "
            "for this district."
        )

        return None


    # IMPORTANT:
    # Use the latest valid row,
    # not simply the last row in the dataset.

    latest = (
        valid_rows
        .sort_values("Date")
        .iloc[-1]
    )


    # -----------------------------------------
    # Create prediction dataframe
    # -----------------------------------------

    prediction_data = pd.DataFrame({

        "nasa_temperature_c":
            [latest["nasa_temperature_c"]],

        "nasa_humidity_percent":
            [latest["nasa_humidity_percent"]],

        "nasa_surface_pressure_kpa":
            [latest["nasa_surface_pressure_kpa"]],

        "nasa_wind_speed_ms":
            [latest["nasa_wind_speed_ms"]],

        "nasa_precipitation_mm":
            [latest["nasa_precipitation_mm"]],

        "imd_daily_rainfall_mm":
            [latest["imd_daily_rainfall_mm"]],

        "previous_day_rainfall_mm":
            [latest["previous_day_rainfall_mm"]],

        "rainfall_3day_avg_mm":
            [latest["rainfall_3day_avg_mm"]],

        "rainfall_7day_avg_mm":
            [latest["rainfall_7day_avg_mm"]],

        "month":
            [latest["month"]],

        "District":
            [latest["District"]],

        "State":
            [latest["State"]]

    })


    # -----------------------------------------
    # Final validation
    # -----------------------------------------

    if prediction_data.isna().any().any():

        print(
            "\nCannot make prediction."
        )

        print(
            "Some weather values are missing."
        )

        return None


    if np.isinf(
        prediction_data.select_dtypes(
            include=np.number
        )
    ).any().any():

        print(
            "\nCannot make prediction."
        )

        print(
            "Some weather values are invalid."
        )

        return None


    # -----------------------------------------
    # Rain probability
    # -----------------------------------------

    rain_prob = (

        rain_model
        .predict_proba(
            prediction_data
        )[0][1]

    )


    # -----------------------------------------
    # Heavy rain probability
    # -----------------------------------------

    heavy_prob = (

        heavy_model
        .predict_proba(
            prediction_data
        )[0][1]

    )


    # -----------------------------------------
    # Weibull expected rainfall
    # -----------------------------------------

    expected_if_wet = (
        weibull_mean
    )


    expected_rainfall = (
        rain_prob
        *
        expected_if_wet
    )


    # -----------------------------------------
    # Probability thresholds
    # -----------------------------------------

    probability_10 = (

        rain_prob
        *
        weibull_probability_above(10)

    )


    probability_20 = (

        rain_prob
        *
        weibull_probability_above(20)

    )


    probability_30 = (

        rain_prob
        *
        weibull_probability_above(30)

    )


    probability_50 = (

        rain_prob
        *
        weibull_probability_above(50)

    )


    # -----------------------------------------
    # Category
    # -----------------------------------------

    if rain_prob < 0.30:

        category = (
            "Low chance of rain"
        )

    elif rain_prob < 0.60:

        category = (
            "Moderate chance of rain"
        )

    elif rain_prob < 0.80:

        category = (
            "High chance of rain"
        )

    else:

        category = (
            "Very high chance of rain"
        )


    # -----------------------------------------
    # Prediction date
    # -----------------------------------------

    prediction_date = (
        latest["Date"]
        +
        pd.Timedelta(days=1)
    )


    # -----------------------------------------
    # Display result
    # -----------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "RAINFALL PREDICTION"
    )

    print("=" * 70)


    print(
        "\nDistrict:",
        latest["District"]
    )


    print(
        "State:",
        latest["State"]
    )


    print(
        "Latest VALID data date:",
        latest["Date"].date()
    )


    print(
        "Prediction for:",
        prediction_date.date()
    )


    print("\nCURRENT WEATHER")


    print(
        "Temperature:",
        round(
            latest[
                "nasa_temperature_c"
            ],
            2
        ),
        "°C"
    )


    print(
        "Humidity:",
        round(
            latest[
                "nasa_humidity_percent"
            ],
            2
        ),
        "%"
    )


    print(
        "Pressure:",
        round(
            latest[
                "nasa_surface_pressure_kpa"
            ],
            2
        ),
        "kPa"
    )


    print(
        "Wind:",
        round(
            latest[
                "nasa_wind_speed_ms"
            ],
            2
        ),
        "m/s"
    )


    print(
        "Today's rainfall:",
        round(
            latest[
                "imd_daily_rainfall_mm"
            ],
            2
        ),
        "mm"
    )


    print(
        "\n" + "-" * 70
    )


    print(
        "PREDICTION FOR NEXT DAY"
    )


    print(
        "-" * 70
    )


    print(
        "\nProbability of rain:",
        f"{rain_prob * 100:.2f}%"
    )


    print(
        "Probability of >20 mm:",
        f"{heavy_prob * 100:.2f}%"
    )


    print(
        "\nExpected rainfall if it rains:",
        f"{expected_if_wet:.2f} mm"
    )


    print(
        "Expected rainfall:",
        f"{expected_rainfall:.2f} mm"
    )


    print(
        "\nProbability >10 mm:",
        f"{probability_10 * 100:.2f}%"
    )


    print(
        "Probability >20 mm:",
        f"{probability_20 * 100:.2f}%"
    )


    print(
        "Probability >30 mm:",
        f"{probability_30 * 100:.2f}%"
    )


    print(
        "Probability >50 mm:",
        f"{probability_50 * 100:.2f}%"
    )


    print(
        "\nRainfall outlook:",
        category
    )


    print("=" * 70)


    return {

        "district":
            latest["District"],

        "state":
            latest["State"],

        "latest_valid_date":
            latest["Date"],

        "prediction_date":
            prediction_date,

        "rain_probability":
            rain_prob,

        "heavy_rain_probability":
            heavy_prob,

        "expected_rainfall_mm":
            expected_rainfall,

        "probability_above_10mm":
            probability_10,

        "probability_above_20mm":
            probability_20,

        "probability_above_30mm":
            probability_30,

        "probability_above_50mm":
            probability_50

    }


# ============================================================
# 27. INTERACTIVE DISTRICT SELECTION
# ============================================================

print("\n" + "=" * 70)

print(
    "INTERACTIVE DISTRICT PREDICTION"
)

print("=" * 70)


district_input = input(
    "\nEnter district name "
    "(or type 'exit'): "
)


if district_input.lower().strip() != "exit":

    predict_for_district(
        district_input
    )


print("\nProgram finished.")
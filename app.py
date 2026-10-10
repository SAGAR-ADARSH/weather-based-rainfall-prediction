from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DATA_FILE = "combined_rainfall_weather.csv"
RAIN_THRESHOLD = 20.0

app = Flask(__name__)

# -----------------------------
# Load and prepare data
# -----------------------------
df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# Treat NASA error placeholders as missing
df = df.replace(
    [-999, -999.0, -9999, -9999.0, np.inf, -np.inf],
    np.nan
)

df = df.sort_values(["District", "Date"]).reset_index(drop=True)

rainfall_col = "imd_daily_rainfall_mm"

# Rebuild temporal features to guarantee correct ordering
df["previous_day_rainfall_mm"] = (
    df.groupby("District")[rainfall_col].shift(1)
)
df["rainfall_3day_avg_mm"] = (
    df.groupby("District")[rainfall_col]
      .transform(lambda x: x.rolling(3, min_periods=1).mean())
)
df["rainfall_7day_avg_mm"] = (
    df.groupby("District")[rainfall_col]
      .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
df["next_day_rainfall_mm"] = (
    df.groupby("District")[rainfall_col].shift(-1)
)

df["next_day_rain"] = np.where(
    df["next_day_rainfall_mm"].notna(),
    (df["next_day_rainfall_mm"] > 0).astype(int),
    np.nan
)
df["next_day_heavy_rain"] = np.where(
    df["next_day_rainfall_mm"].notna(),
    (df["next_day_rainfall_mm"] > RAIN_THRESHOLD).astype(int),
    np.nan
)

df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["day_of_year"] = df["Date"].dt.dayofyear

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
categorical_features = ["District", "State"]

numeric_features = [x for x in numeric_features if x in df.columns]
categorical_features = [x for x in categorical_features if x in df.columns]

model_df = df[
    ["Date"] + numeric_features + categorical_features +
    ["next_day_rainfall_mm", "next_day_rain", "next_day_heavy_rain"]
].copy()

model_df = model_df.replace(
    [-999, -999.0, -9999, -9999.0, np.inf, -np.inf],
    np.nan
)
model_df = model_df.dropna(
    subset=numeric_features + categorical_features +
           ["next_day_rainfall_mm", "next_day_rain", "next_day_heavy_rain"]
)

# -----------------------------
# Chronological training split
# -----------------------------
dates = sorted(model_df["Date"].unique())
split = max(1, int(len(dates) * 0.80))
train_dates = dates[:split]
test_dates = dates[split:]

train_df = model_df[model_df["Date"].isin(train_dates)]
test_df = model_df[model_df["Date"].isin(test_dates)]

X_train = train_df[numeric_features + categorical_features]

y_train_rain = train_df["next_day_rain"].astype(int)
y_train_heavy = train_df["next_day_heavy_rain"].astype(int)

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

rain_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", LogisticRegression(max_iter=3000))
])
rain_model.fit(X_train, y_train_rain)

heavy_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", LogisticRegression(max_iter=3000))
])
heavy_model.fit(X_train, y_train_heavy)

# -----------------------------
# Global Weibull distribution
# -----------------------------
positive_rainfall = (
    df.loc[df[rainfall_col] > 0, rainfall_col]
      .dropna()
      .values
)

if len(positive_rainfall):
    weibull_shape, _, weibull_scale = stats.weibull_min.fit(
        positive_rainfall, floc=0
    )
    weibull_mean = stats.weibull_min.mean(
        weibull_shape, loc=0, scale=weibull_scale
    )
else:
    weibull_shape, weibull_scale, weibull_mean = 1.0, 1.0, 0.0

def weibull_sf(threshold):
    return float(stats.weibull_min.sf(
        threshold,
        weibull_shape,
        loc=0,
        scale=weibull_scale
    ))

# -----------------------------
# Helpers
# -----------------------------
def clean_name(value):
    return str(value).strip().lower()

def find_district(name, state=None):
    key = clean_name(name)
    matches = df[df["District"].astype(str).str.strip().str.lower() == key]
    if state:
        state_key = clean_name(state)
        matches = matches[matches["State"].astype(str).str.strip().str.lower() == state_key]
    return matches.copy()

def predict_for_district(name, state=None):
    district_data = find_district(name, state)

    if district_data.empty:
        return {"error": "District not found."}

    valid_rows = district_data.dropna(
        subset=numeric_features + categorical_features
    ).copy()

    valid_rows = valid_rows[
        ~valid_rows[numeric_features].isin(
            [-999, -999.0, -9999, -9999.0]
        ).any(axis=1)
    ]

    if valid_rows.empty:
        return {
            "error": "No valid weather data is available for this district."
        }

    latest = valid_rows.sort_values("Date").iloc[-1]

    X = pd.DataFrame({
        feature: [latest[feature]]
        for feature in numeric_features + categorical_features
    })

    rain_prob = float(rain_model.predict_proba(X)[0, 1])
    heavy_prob = float(heavy_model.predict_proba(X)[0, 1])

    expected_if_wet = float(weibull_mean)
    expected_rainfall = rain_prob * expected_if_wet

    probabilities = {
        "10": rain_prob * weibull_sf(10),
        "20": rain_prob * weibull_sf(20),
        "30": rain_prob * weibull_sf(30),
        "50": rain_prob * weibull_sf(50),
    }

    if rain_prob < 0.30:
        category = "Low chance of rain"
    elif rain_prob < 0.60:
        category = "Moderate chance of rain"
    elif rain_prob < 0.80:
        category = "High chance of rain"
    else:
        category = "Very high chance of rain"

    prediction_date = latest["Date"] + pd.Timedelta(days=1)

    return {
        "district": str(latest["District"]),
        "state": str(latest["State"]),
        "latest_valid_date": latest["Date"].strftime("%Y-%m-%d"),
        "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        "weather": {
            "temperature": float(latest["nasa_temperature_c"]),
            "humidity": float(latest["nasa_humidity_percent"]),
            "pressure": float(latest["nasa_surface_pressure_kpa"]),
            "wind": float(latest["nasa_wind_speed_ms"]),
            "today_rainfall": float(latest["imd_daily_rainfall_mm"]),
        },
        "prediction": {
            "rain_probability": rain_prob * 100,
            "heavy_rain_probability": heavy_prob * 100,
            "expected_if_wet": expected_if_wet,
            "expected_rainfall": expected_rainfall,
            "above_10": probabilities["10"] * 100,
            "above_20": probabilities["20"] * 100,
            "above_30": probabilities["30"] * 100,
            "above_50": probabilities["50"] * 100,
            "category": category,
        },
        "model": {
            "weibull_shape": float(weibull_shape),
            "weibull_scale": float(weibull_scale),
            "training_rows": int(len(train_df)),
            "districts": int(df["District"].nunique()),
            "states": int(df["State"].nunique()),
        }
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    districts = sorted(
        df["District"].dropna().astype(str).unique().tolist()
    )
    return render_template("index.html", districts=districts)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    district = data.get("district", "")
    state = data.get("state") or None
    if not district.strip():
        return jsonify({"error": "Please select or enter a district."}), 400

    result = predict_for_district(district, state)
    status = 200 if "error" not in result else 404
    return jsonify(result), status

@app.route("/api/info")
def api_info():
    return jsonify({
        "districts": int(df["District"].nunique()),
        "states": int(df["State"].nunique()),
        "rows": int(len(df)),
        "date_start": df["Date"].min().strftime("%Y-%m-%d"),
        "date_end": df["Date"].max().strftime("%Y-%m-%d"),
        "weibull_shape": float(weibull_shape),
        "weibull_scale": float(weibull_scale),
        "wet_day_mean": float(weibull_mean),
    })

# -----------------------------
# Map support (additive; does not change the model)
# -----------------------------
COORD_FILE = "district_coordinates.csv"

def load_district_coordinates():
    try:
        coords = pd.read_csv(COORD_FILE)
    except (OSError, ValueError):
        return {}
    coords = coords.dropna(subset=["State", "District"])
    return {
        (clean_name(r.State), clean_name(r.District)): (
            float(r.Latitude), float(r.Longitude)
        )
        for r in coords.itertuples(index=False)
        if pd.notna(r.Latitude) and pd.notna(r.Longitude)
    }

DISTRICT_COORDS = load_district_coordinates()

@app.route("/api/districts")
def api_districts():
    pairs = (
        df[["State", "District"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["State", "District"])
    )
    items = []
    for row in pairs.itertuples(index=False):
        coord = DISTRICT_COORDS.get((clean_name(row.State), clean_name(row.District)))
        items.append({
            "district": str(row.District),
            "state": str(row.State),
            "latitude": coord[0] if coord else None,
            "longitude": coord[1] if coord else None,
        })
    return jsonify({"districts": items})

@app.route("/api/history", methods=["GET"])
def api_history():
    """Real observed daily rainfall for one district (not simulated)."""
    name = request.args.get("district", "")
    state = request.args.get("state")
    try:
        days = max(7, min(365, int(request.args.get("days", 60))))
    except ValueError:
        days = 60

    if not name.strip():
        return jsonify({"error": "Please select a district."}), 400

    data = find_district(name, state)
    if data.empty:
        return jsonify({"error": "District not found."}), 404

    data = data.sort_values("Date").tail(days)
    rows = [
        {
            "date": r.Date.strftime("%Y-%m-%d"),
            "rainfall_mm": None if pd.isna(r.imd_daily_rainfall_mm)
            else float(r.imd_daily_rainfall_mm),
        }
        for r in data.itertuples(index=False)
    ]
    return jsonify({
        "district": str(data["District"].iloc[0]),
        "state": str(data["State"].iloc[0]),
        "source": "IMD daily rainfall observations in combined_rainfall_weather.csv",
        "days": len(rows),
        "observations": rows,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

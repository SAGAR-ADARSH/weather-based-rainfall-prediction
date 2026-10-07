import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

IMD_FILE = "rainfall_districtwise_daily_imd.csv"
COORD_FILE = "district_coordinates.csv"
OUTPUT_FILE = "combined_rainfall_weather.csv"

# NASA recommends keeping the number of simultaneous
# requests reasonably low.
MAX_WORKERS = 5

# NASA POWER variables
NASA_PARAMETERS = [
    "T2M",           # Temperature at 2m (°C)
    "RH2M",          # Relative Humidity at 2m (%)
    "PS",            # Surface Pressure (kPa)
    "WS10M",         # Wind Speed at 10m (m/s)
    "PRECTOTCORR"    # Corrected precipitation (mm/day)
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.lower()
    )

    return df


def find_column(df, possible_names):

    for name in possible_names:

        name = name.lower()

        if name in df.columns:
            return name

    raise ValueError(
        f"\nCould not find any of these columns:\n"
        f"{possible_names}\n\n"
        f"Available columns are:\n"
        f"{list(df.columns)}"
    )


def normalize_text(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def create_session():

    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount(
        "https://",
        adapter
    )

    session.mount(
        "http://",
        adapter
    )

    return session


# ============================================================
# 1. LOAD IMD DATA
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING IMD DATA")
print("=" * 70)

imd = pd.read_csv(IMD_FILE)

imd = clean_columns(imd)

print("\nIMD shape:", imd.shape)

print("\nIMD columns:")
for col in imd.columns:
    print("  ", col)


# ============================================================
# 2. IDENTIFY IMD COLUMNS
# ============================================================

state_col = find_column(
    imd,
    [
        "state"
    ]
)

district_col = find_column(
    imd,
    [
        "district"
    ]
)

date_col = find_column(
    imd,
    [
        "date"
    ]
)

rainfall_col = find_column(
    imd,
    [
        "daily_actual",
        "daily_actual_mm",
        "rainfall",
        "rainfall_mm"
    ]
)


print("\nDetected columns:")

print("State    :", state_col)
print("District :", district_col)
print("Date     :", date_col)
print("Rainfall :", rainfall_col)


# ============================================================
# 3. CLEAN IMD DATA
# ============================================================

imd[date_col] = pd.to_datetime(
    imd[date_col],
    errors="coerce"
)

imd[rainfall_col] = pd.to_numeric(
    imd[rainfall_col],
    errors="coerce"
)

# Remove invalid dates

imd = imd.dropna(
    subset=[date_col]
)

print("\nIMD date range:")

print(
    imd[date_col].min(),
    "to",
    imd[date_col].max()
)

print(
    "\nNumber of states:",
    imd[state_col].nunique()
)

print(
    "Number of districts:",
    imd[district_col].nunique()
)

print(
    "Number of dates:",
    imd[date_col].nunique()
)


# ============================================================
# 4. LOAD DISTRICT COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING DISTRICT COORDINATES")
print("=" * 70)

coords = pd.read_csv(
    COORD_FILE
)

coords = clean_columns(
    coords
)

print("\nCoordinate shape:", coords.shape)

print("\nCoordinate columns:")

for col in coords.columns:
    print("  ", col)


# ============================================================
# 5. IDENTIFY COORDINATE COLUMNS
# ============================================================

coord_state_col = find_column(
    coords,
    [
        "state"
    ]
)

coord_district_col = find_column(
    coords,
    [
        "district"
    ]
)

lat_col = find_column(
    coords,
    [
        "latitude",
        "lat"
    ]
)

lon_col = find_column(
    coords,
    [
        "longitude",
        "lon",
        "lng"
    ]
)


print("\nDetected coordinate columns:")

print(
    "State     :",
    coord_state_col
)

print(
    "District  :",
    coord_district_col
)

print(
    "Latitude  :",
    lat_col
)

print(
    "Longitude :",
    lon_col
)


# ============================================================
# 6. CLEAN COORDINATES
# ============================================================

coords[lat_col] = pd.to_numeric(
    coords[lat_col],
    errors="coerce"
)

coords[lon_col] = pd.to_numeric(
    coords[lon_col],
    errors="coerce"
)


# ============================================================
# 7. NORMALIZE STATE AND DISTRICT NAMES
# ============================================================

imd["_state_key"] = normalize_text(
    imd[state_col]
)

imd["_district_key"] = normalize_text(
    imd[district_col]
)

coords["_state_key"] = normalize_text(
    coords[coord_state_col]
)

coords["_district_key"] = normalize_text(
    coords[coord_district_col]
)


# ============================================================
# 8. CHECK DISTRICT MATCHING
# ============================================================

print("\n" + "=" * 70)
print("3. CHECKING DISTRICT COORDINATE MATCHING")
print("=" * 70)


imd_locations = (
    imd[
        [
            "_state_key",
            "_district_key"
        ]
    ]
    .drop_duplicates()
)


coord_locations = (
    coords[
        [
            "_state_key",
            "_district_key"
        ]
    ]
    .drop_duplicates()
)


location_check = imd_locations.merge(
    coord_locations,
    on=[
        "_state_key",
        "_district_key"
    ],
    how="left",
    indicator=True
)


missing_locations = location_check[
    location_check["_merge"] == "left_only"
]


print(
    "\nIMD unique locations:",
    len(imd_locations)
)

print(
    "Coordinate locations:",
    len(coord_locations)
)

print(
    "Unmatched locations:",
    len(missing_locations)
)


if len(missing_locations) > 0:

    print(
        "\nWARNING: These locations have no coordinate match:"
    )

    print(
        missing_locations[
            [
                "_state_key",
                "_district_key"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 9. MERGE COORDINATES WITH IMD
# ============================================================

print("\nMerging coordinates with IMD...")

imd = imd.merge(

    coords[
        [
            "_state_key",
            "_district_key",
            lat_col,
            lon_col
        ]
    ],

    on=[
        "_state_key",
        "_district_key"
    ],

    how="left"
)


rows_before = len(imd)


imd = imd.dropna(
    subset=[
        lat_col,
        lon_col
    ]
)


rows_after = len(imd)


print(
    "\nIMD rows before coordinate filtering:",
    rows_before
)

print(
    "IMD rows after coordinate filtering:",
    rows_after
)

print(
    "Rows removed:",
    rows_before - rows_after
)


# ============================================================
# 10. DETERMINE NASA DATE RANGE
# ============================================================

start_date = imd[
    date_col
].min().strftime("%Y%m%d")


end_date = imd[
    date_col
].max().strftime("%Y%m%d")


print("\n" + "=" * 70)
print("4. NASA POWER DATE RANGE")
print("=" * 70)

print(
    "\nStart:",
    start_date
)

print(
    "End:",
    end_date
)


# ============================================================
# 11. PREPARE DISTRICT LIST
# ============================================================

locations = (

    imd[
        [
            "_state_key",
            "_district_key",
            state_col,
            district_col,
            lat_col,
            lon_col
        ]
    ]

    .drop_duplicates(
        subset=[
            "_state_key",
            "_district_key"
        ]
    )

    .reset_index(drop=True)

)


print(
    "\nDistricts to download:",
    len(locations)
)


# ============================================================
# 12. NASA POWER DOWNLOAD FUNCTION
# ============================================================

def download_nasa(row):

    state = row[state_col]

    district = row[district_col]

    latitude = float(
        row[lat_col]
    )

    longitude = float(
        row[lon_col]
    )


    url = (
        "https://power.larc.nasa.gov/api/"
        "temporal/daily/point"
    )


    parameters = ",".join(
        NASA_PARAMETERS
    )


    params = {

        "parameters":
            parameters,

        "community":
            "AG",

        "longitude":
            longitude,

        "latitude":
            latitude,

        "start":
            start_date,

        "end":
            end_date,

        "format":
            "JSON",

        "header":
            "true",

        "time-standard":
            "LST"

    }


    session = create_session()


    try:

        response = session.get(

            url,

            params=params,

            timeout=60

        )


        response.raise_for_status()


        data = response.json()


        parameter_data = (
            data
            .get("properties", {})
            .get("parameter", {})
        )


        if not parameter_data:

            raise ValueError(
                "NASA returned no parameter data"
            )


        # ----------------------------------------
        # Find all available dates
        # ----------------------------------------

        dates = set()


        for parameter in NASA_PARAMETERS:

            if parameter in parameter_data:

                dates.update(
                    parameter_data[
                        parameter
                    ].keys()
                )


        if not dates:

            raise ValueError(
                "No dates returned by NASA"
            )


        nasa_df = pd.DataFrame(
            {
                "date":
                    sorted(dates)
            }
        )


        # ----------------------------------------
        # Add NASA variables
        # ----------------------------------------

        for parameter in NASA_PARAMETERS:

            values = parameter_data.get(
                parameter,
                {}
            )

            nasa_df[parameter] = (
                nasa_df["date"]
                .map(values)
            )


        # ----------------------------------------
        # Convert date
        # ----------------------------------------

        nasa_df["date"] = pd.to_datetime(
            nasa_df["date"],
            format="%Y%m%d",
            errors="coerce"
        )


        # ----------------------------------------
        # Add location information
        # ----------------------------------------

        nasa_df["state"] = state

        nasa_df["district"] = district

        nasa_df["latitude"] = latitude

        nasa_df["longitude"] = longitude


        return nasa_df


    except Exception as e:

        print(
            f"\nNASA ERROR:"
            f" {state} - {district}"
        )

        print(
            "Reason:",
            e
        )

        return None


# ============================================================
# 13. DOWNLOAD NASA DATA
# ============================================================

print("\n" + "=" * 70)
print("5. DOWNLOADING NASA POWER DATA")
print("=" * 70)

print(
    "\nUsing",
    MAX_WORKERS,
    "parallel requests."
)

print(
    "This may take several minutes."
)


results = []


with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:


    futures = {

        executor.submit(
            download_nasa,
            row
        ):
        index

        for index, row
        in locations.iterrows()

    }


    for future in tqdm(

        as_completed(futures),

        total=len(futures),

        desc="Downloading districts"

    ):


        result = future.result()


        if result is not None:

            results.append(
                result
            )


# ============================================================
# 14. CHECK NASA RESULTS
# ============================================================

if not results:

    raise RuntimeError(
        "\nNASA POWER download failed "
        "for every district."
    )


nasa = pd.concat(
    results,
    ignore_index=True
)


print(
    "\nNASA rows downloaded:",
    len(nasa)
)


print(
    "NASA districts downloaded:",
    nasa["district"].nunique()
)


# ============================================================
# 15. PREPARE NASA DATA
# ============================================================

nasa = nasa.rename(
    columns={

        "state":
            "_state_key",

        "district":
            "_district_key",

        "T2M":
            "nasa_temperature_c",

        "RH2M":
            "nasa_humidity_percent",

        "PS":
            "nasa_surface_pressure_kpa",

        "WS10M":
            "nasa_wind_speed_ms",

        "PRECTOTCORR":
            "nasa_precipitation_mm"

    }
)


nasa["_state_key"] = normalize_text(
    nasa["_state_key"]
)

nasa["_district_key"] = normalize_text(
    nasa["_district_key"]
)


# ============================================================
# 16. MERGE IMD + NASA
# ============================================================

print("\n" + "=" * 70)
print("6. MERGING IMD + NASA POWER")
print("=" * 70)


imd = imd.rename(
    columns={
        date_col:
            "date"
    }
)


combined = imd.merge(

    nasa,

    on=[
        "_state_key",
        "_district_key",
        "date"
    ],

    how="left"

)


print(
    "\nCombined dataset shape:",
    combined.shape
)


# ============================================================
# 17. SORT DATA
# ============================================================

combined = combined.sort_values(
    [
        "_district_key",
        "date"
    ]
)


# ============================================================
# 18. CREATE TIME FEATURES
# ============================================================

print("\n" + "=" * 70)
print("7. CREATING FEATURES")
print("=" * 70)


combined["month"] = (
    combined["date"].dt.month
)


combined["day"] = (
    combined["date"].dt.day
)


combined["day_of_year"] = (
    combined["date"].dt.dayofyear
)


# ============================================================
# 19. CREATE SEASON
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:

        return "Winter"

    elif month in [3, 4, 5]:

        return "Summer"

    elif month in [6, 7, 8, 9]:

        return "Monsoon"

    else:

        return "Post-Monsoon"


combined["season"] = (
    combined["month"]
    .apply(get_season)
)


# ============================================================
# 20. PREVIOUS DAY RAINFALL
# ============================================================

combined[
    "previous_day_rainfall_mm"
] = (

    combined
    .groupby("_district_key")[
        rainfall_col
    ]
    .shift(1)

)


# ============================================================
# 21. THREE DAY RAINFALL AVERAGE
# ============================================================

combined[
    "rainfall_3day_avg_mm"
] = (

    combined
    .groupby("_district_key")[
        rainfall_col
    ]
    .transform(

        lambda x:

        x.shift(1)
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()

    )

)


# ============================================================
# 22. SEVEN DAY RAINFALL AVERAGE
# ============================================================

combined[
    "rainfall_7day_avg_mm"
] = (

    combined
    .groupby("_district_key")[
        rainfall_col
    ]
    .transform(

        lambda x:

        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()

    )

)


# ============================================================
# 23. CREATE TOMORROW'S RAINFALL TARGET
# ============================================================

combined[
    "next_day_rainfall_mm"
] = (

    combined
    .groupby("_district_key")[
        rainfall_col
    ]
    .shift(-1)

)


# ============================================================
# 24. CREATE RAIN / NO RAIN TARGET
# ============================================================

combined[
    "next_day_rain"
] = (

    combined[
        "next_day_rainfall_mm"
    ]

    > 0

).astype("Int64")


# ============================================================
# 25. RENAME IMD RAINFALL
# ============================================================

combined = combined.rename(
    columns={

        rainfall_col:
            "imd_daily_rainfall_mm"

    }
)


# ============================================================
# 26. REMOVE INTERNAL COLUMNS
# ============================================================

combined = combined.drop(

    columns=[
        "_state_key",
        "_district_key"
    ],

    errors="ignore"

)


# ============================================================
# 27. REMOVE DUPLICATE COLUMNS
# ============================================================

combined = combined.loc[
    :,
    ~combined.columns.duplicated()
]


# ============================================================
# 28. RENAME FINAL COLUMNS
# ============================================================

combined = combined.rename(
    columns={

        "date":
            "Date",

        "state":
            "State",

        "district":
            "District",

        "latitude":
            "Latitude",

        "longitude":
            "Longitude"

    }
)


# ============================================================
# 29. REORDER IMPORTANT COLUMNS
# ============================================================

preferred_columns = [

    "Date",

    "State",

    "District",

    "Latitude",

    "Longitude",

    "imd_daily_rainfall_mm",

    "nasa_temperature_c",

    "nasa_humidity_percent",

    "nasa_surface_pressure_kpa",

    "nasa_wind_speed_ms",

    "nasa_precipitation_mm",

    "month",

    "day",

    "day_of_year",

    "season",

    "previous_day_rainfall_mm",

    "rainfall_3day_avg_mm",

    "rainfall_7day_avg_mm",

    "next_day_rainfall_mm",

    "next_day_rain"

]


existing_columns = [

    column

    for column
    in preferred_columns

    if column in combined.columns

]


other_columns = [

    column

    for column
    in combined.columns

    if column not in existing_columns

]


combined = combined[
    existing_columns +
    other_columns
]


# ============================================================
# 30. SAVE DATASET
# ============================================================

combined.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# 31. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("                 COMBINATION COMPLETE")
print("=" * 70)


print(
    "\nOutput file:",
    OUTPUT_FILE
)


print(
    "\nTotal rows:",
    len(combined)
)


print(
    "Total columns:",
    len(combined.columns)
)


print(
    "States:",
    combined["State"].nunique()
)


print(
    "Districts:",
    combined["District"].nunique()
)


print(
    "Dates:",
    combined["Date"].nunique()
)


print(
    "\nDate range:"
)


print(
    combined["Date"].min(),
    "to",
    combined["Date"].max()
)


# ============================================================
# 32. NASA MISSING DATA REPORT
# ============================================================

print("\n" + "=" * 70)
print("NASA DATA QUALITY")
print("=" * 70)


nasa_columns = [

    "nasa_temperature_c",

    "nasa_humidity_percent",

    "nasa_surface_pressure_kpa",

    "nasa_wind_speed_ms",

    "nasa_precipitation_mm"

]


for column in nasa_columns:

    if column in combined.columns:

        missing = (
            combined[column]
            .isna()
            .sum()
        )


        percentage = (

            missing /
            len(combined)

        ) * 100


        print(

            f"{column}: "
            f"{missing} missing "
            f"({percentage:.2f}%)"

        )


# ============================================================
# 33. PREDICTION TARGET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION TARGET")
print("=" * 70)


print(
    "\nTarget 1:"
)

print(
    "next_day_rain"
)

print(
    "Used for rain/no-rain prediction."
)


print(
    "\nTarget 2:"
)

print(
    "next_day_rainfall_mm"
)

print(
    "Used for rainfall amount prediction."
)


# ============================================================
# 34. DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)


print(
    combined
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 35. FINISHED
# ============================================================

print("\n" + "=" * 70)

print(
    f"Dataset saved successfully:"
    f" {OUTPUT_FILE}"
)

print("=" * 70)
# RainPredict — Weather-Based Rainfall Prediction Website

This is a Flask website that embeds the global rainfall prediction model.

## Folder structure

```text
rainfall_prediction_website/
├── app.py
├── combined_rainfall_weather.csv   <-- put your dataset here
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Setup

Inside your existing project environment:

```bash
cd "/home/adarsh/Desktop/PROJECT PROBABILITY"
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `app.py`, `templates/`, `static/`, and `requirements.txt` into your project folder.

Make sure this file is present beside `app.py`:

```text
combined_rainfall_weather.csv
```

## Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How it works

1. Loads the combined rainfall + NASA weather dataset.
2. Cleans `-999`/`-9999` missing-value placeholders.
3. Sorts each district chronologically.
4. Creates next-day rainfall targets.
5. Trains a global Logistic Regression rain/no-rain model.
6. Trains a global Logistic Regression heavy-rain model (>20 mm).
7. Fits a Weibull distribution to positive rainfall.
8. Exposes `/api/predict`.
9. The website sends a selected district to the API and displays the prediction.

## Important academic limitation

The current dataset contains a short time period, so this should be presented as an academic prototype rather than a production weather forecasting system. More years of daily data would substantially improve the model.

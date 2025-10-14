import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from pathlib import Path
import os
from datetime import datetime

app = Flask(__name__)

# try multiple likely locations for the trained model file
base = Path(__file__).parent
candidates = [
    base / "pumpkin_model.pkl",
    base / "models" / "pumpkin_model.pkl",
    base.parent / "pumpkin_model.pkl",
    base.parent.parent / "pumpkin_model.pkl",
    Path.cwd() / "pumpkin_model.pkl",
    Path.cwd() / "3-Web-App" / "1-Web-App" / "pumpkin_model.pkl",
    Path.cwd() / "2-Regression" / "4-Logistic" / "pumpkin_model.pkl",
    Path.cwd() / "1-Web-App" / "pumpkin_model.pkl",
    base / "solution" / "pumpkin_model.pkl",
]

model = None
model_path = None
for p in candidates:
    try:
        p = p.resolve()
    except Exception:
        pass
    if p.exists():
        with open(p, "rb") as f:
            model = pickle.load(f)
        model_path = p
        break

if model is None:
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"pumpkin_model.pkl not found. Searched: {searched}\n"
        "Place pumpkin_model.pkl in one of the searched locations or update the path."
    )

# attempt to load a saved label encoder (optional)
label_encoder = None
for p in [model_path.parent / "label_encoder.pkl", base / "label_encoder.pkl"]:
    if p.exists():
        try:
            with open(p, "rb") as f:
                label_encoder = pickle.load(f)
            break
        except Exception:
            label_encoder = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # map form fields -> dataset column names used at training
    # form provides: city_name, package, variety, item_size
    form = request.form
    now = datetime.now()
    # expected raw columns from the notebook/data prep step
    expected_cols = [
        "City Name",
        "Package",
        "Variety",
        "Low Price",
        "High Price",
        "Mostly Low",
        "Mostly High",
        "Origin",
        "Item Size",
        "Repack",
        "Year",
        "Month",
        "Day",
    ]

    # fill row with values from form when available, otherwise sensible defaults
    row = {
        "City Name": form.get("city_name", "").strip() or "Unknown",
        "Package": form.get("package", "").strip() or "Unknown",
        "Variety": form.get("variety", "").strip() or "Unknown",
        # numeric fields - defaults to 0.0
        "Low Price": _try_parse_float(form.get("low_price"), 0.0),
        "High Price": _try_parse_float(form.get("high_price"), 0.0),
        "Mostly Low": _try_parse_float(form.get("mostly_low"), 0.0),
        "Mostly High": _try_parse_float(form.get("mostly_high"), 0.0),
        "Origin": form.get("origin", "").strip() or "US",
        "Item Size": form.get("item_size", "").strip() or "med",
        "Repack": form.get("repack", "").strip() or "No",
        "Year": _try_parse_int(form.get("year"), now.year),
        "Month": _try_parse_int(form.get("month"), now.month),
        "Day": _try_parse_int(form.get("day"), now.day),
    }

    # build DataFrame for prediction (one row)
    X_input = pd.DataFrame([row], columns=expected_cols)

    # Try predict using the model. Different saved models expect different input types:
    # - If the model is a pipeline that includes preprocessing it will accept the raw DataFrame.
    # - Otherwise it may expect a numeric array; we attempt several fallbacks.
    try:
        pred = model.predict(X_input)
    except Exception:
        try:
            # maybe model expects ndarray of numeric values (already encoded)
            pred = model.predict(X_input.values)
        except Exception as e:
            # final fallback: try to construct a small numeric vector from available numeric fields
            numeric_vals = [
                row["Low Price"],
                row["High Price"],
                row["Mostly Low"],
                row["Mostly High"],
                row["Year"],
                row["Month"],
                row["Day"],
            ]
            try:
                pred = model.predict([numeric_vals])
            except Exception as e2:
                # give the user a helpful error message in the page
                return render_template(
                    "index.html",
                    prediction_text=(
                        "Prediction failed: model couldn't handle input. "
                        "Ensure pumpkin_model.pkl contains preprocessing (or provide all numeric inputs). "
                        f"Error: {str(e2)}"
                    ),
                )

    # decode prediction to human-readable color
    # pred may be array of labels (str or numeric)
    output_label = pred[0]

    # if label_encoder available, inverse-transform
    if label_encoder is not None:
        try:
            color = label_encoder.inverse_transform([output_label])[0]
            color = str(color)
        except Exception:
            color = str(output_label)
    else:
        # if model.classes_ are strings, and output_label is index, try to map
        if hasattr(model, "classes_") and isinstance(model.classes_, (list, np.ndarray)):
            try:
                classes = list(model.classes_)
                # if output_label is an index (int) and in range, map it
                if isinstance(output_label, (int, np.integer)) and 0 <= int(output_label) < len(classes):
                    color = str(classes[int(output_label)])
                else:
                    color = str(output_label)
            except Exception:
                color = str(output_label)
        else:
            # fallback mapping used in notebook (may need adjustment)
            fallback = {0: "ORANGE", 1: "WHITE"}
            color = fallback.get(int(output_label), str(output_label))

    return render_template("index.html", prediction_text=f"Likely color: {color}")


def _try_parse_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _try_parse_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


if __name__ == "__main__":
    app.run(debug=True)
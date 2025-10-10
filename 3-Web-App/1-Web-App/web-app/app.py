import numpy as np
from flask import Flask, request, render_template
import pickle
from pathlib import Path
import os

app = Flask(__name__)

# replace single-path load with a search for common locations
base = Path(__file__).parent
candidates = [
    base / "ufo-model.pkl",
    base / "models" / "ufo-model.pkl",
    base.parent / "ufo-model.pkl",
    Path.cwd() / "ufo-model.pkl",
    Path.cwd() / "3-Web-App" / "1-Web-App" / "solution" / "ufo-model.pkl",
]

model = None
for p in candidates:
    if p.exists():
        with open(p, "rb") as f:
            model = pickle.load(f)
        break

if model is None:
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"ufo-model.pkl not found. Searched: {searched}\n"
        "Place ufo-model.pkl in the web-app folder or update the path."
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
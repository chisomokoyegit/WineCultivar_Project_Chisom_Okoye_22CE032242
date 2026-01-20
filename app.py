from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model/wine_cultivar_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["ash"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["total_phenols"]),
            float(request.form["proline"])
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        result = f"Cultivar {prediction + 1}"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

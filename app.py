from pathlib import Path
import json

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED = {"jpg", "jpeg", "png"}
MODEL_PATH = Path("model/plantcare_final.keras")
CLASS_PATH = Path("model/class_names.json")

model = None
class_names = None

def load_model_if_available():
    global model, class_names
    if model is None and MODEL_PATH.exists() and CLASS_PATH.exists():
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = json.loads(CLASS_PATH.read_text())

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No image was uploaded."}), 400

    if not allowed(file.filename):
        return jsonify({"error": "Please upload a JPG, JPEG or PNG image."}), 400

    load_model_if_available()

    if model is None:
        return jsonify({
            "status": "model_not_ready",
            "message": "The website is ready, but the trained crop-disease model has not been installed yet."
        }), 503

    original_name = secure_filename(file.filename)
    path = UPLOAD_DIR / original_name
    file.save(path)

    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as img:
            image = img.convert("RGB").resize((224, 224))

        x = np.asarray(image, dtype=np.float32)[None, ...]
        probabilities = model.predict(x, verbose=0)[0]
        order = np.argsort(probabilities)[::-1]

        top = [
            {"label": class_names[i], "confidence": float(probabilities[i])}
            for i in order[:3]
        ]

        label = top[0]["label"]
        confidence = top[0]["confidence"]

        if confidence < 0.65:
            return jsonify({
                "status": "low_confidence",
                "message": "The image could not be classified reliably. Please upload a clearer leaf photo.",
                "top_predictions": top
            })

        plant, _, disease = label.partition("___")
        return jsonify({
            "status": "ok",
            "plant": plant.replace("_", " "),
            "disease": disease.replace("_", " "),
            "confidence": confidence,
            "top_predictions": top
        })
    finally:
        try:
            path.unlink()
        except OSError:
            pass

if __name__ == "__main__":
    app.run(debug=True)

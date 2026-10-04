import json
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("model/plantcare_final.keras")
class_names = json.loads(open("model/class_names.json").read())

def predict_image(path, threshold=0.65):
    image = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(x, verbose=0)[0]

    order = np.argsort(probabilities)[::-1]
    top = [(class_names[i], float(probabilities[i])) for i in order[:3]]

    label, confidence = top[0]

    if confidence < threshold:
        return {
            "status": "low_confidence",
            "message": "The image could not be classified reliably. Please upload a clearer leaf photo.",
            "top_predictions": top
        }

    plant, _, disease = label.partition("___")
    return {
        "status": "ok",
        "plant": plant.replace("_", " "),
        "disease": disease.replace("_", " "),
        "confidence": confidence,
        "top_predictions": top
    }

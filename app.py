from flask import Flask, request, render_template
import tensorflow as tf
from google.cloud import storage
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

import logging
logging.basicConfig(filename='predictions.log', level=logging.INFO)

# Load model dari Cloud Storage
def load_model_from_gcs():
    storage_client = storage.Client()
    bucket = storage_client.bucket("dog-cat-classification")
    blob = bucket.blob("models/model_v2.h5")
    blob.download_to_filename("model_v2.h5")
    return tf.keras.models.load_model("model_v2.h5")

model = load_model_from_gcs()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file).resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)[0][0]
        result = "Dog" if prediction > 0.5 else "Cat"
        logging.info(f"Prediction: {result}, Confidence: {prediction}")
        return render_template("index.html", prediction=result)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
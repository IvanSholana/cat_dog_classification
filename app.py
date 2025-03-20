from flask import Flask, request, render_template
import tensorflow as tf
from google.cloud import storage
import numpy as np
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model dari Cloud Storage
def load_model_from_gcs():
    try:
        logger.info("Starting to load model from GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket("my-mlops-bucket-2025")
        blob = bucket.blob("models/model.h5")
        logger.info("Downloading model from GCS...")
        blob.download_to_filename("model.h5")
        logger.info("Loading model into TensorFlow...")
        model = tf.keras.models.load_model("model.h5")
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from GCS: {str(e)}")
        raise

try:
    model = load_model_from_gcs()
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            logger.info("Received POST request for prediction")
            file = request.files["file"]
            image = Image.open(file).resize((128, 128))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            prediction = model.predict(image_array)[0][0]
            result = "Dog" if prediction > 0.5 else "Cat"
            logger.info(f"Prediction result: {result}")
            return render_template("index.html", prediction=result)
        return render_template("index.html", prediction=None)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        raise

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
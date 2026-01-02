import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (240, 240)
CLASS_NAMES = ["Cat", "Dog"]

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIR = "/home/site/wwwroot/model"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URL = "https://cnnmodelh5.blob.core.windows.net/cnnmodel/cnn_model.keras?sp=r&st=2026-01-01T18:18:06Z&se=2026-01-30T02:33:06Z&sv=2024-11-04&sr=b&sig=BxQExZck7jJ7wx0SqTOyUchMMg4k%2BulbZdgCMkFZKnQ%3D"

model = None  # 🔴 IMPORTANT

# ---------------- DOWNLOAD MODEL ----------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Azure Blob...")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded")

# ---------------- LAZY LOAD MODEL ----------------
def get_model():
    global model
    if model is None:
        print("🧠 Loading model into memory...")
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded")
    return model

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    mdl = get_model()   # 🔥 LOADS ONLY ON FIRST REQUEST

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    prediction = mdl.predict(img_array)
    score = prediction[0][0]

    label = "Dog" if score >= 0.5 else "Cat"
    confidence = round(max(score, 1 - score) * 100, 2)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run()

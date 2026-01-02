import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.utils import load_img, img_to_array

# ---------------- FORCE CPU (VERY IMPORTANT) ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

# ---------------- FLASK APP ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (240, 240)
CLASS_NAMES = ["Cat", "Dog"]

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL PATH ----------------
MODEL_DIR = "/home/site/wwwroot/model"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URL = "https://cnnmodelh5.blob.core.windows.net/cnnmodel/cnn_model_tf215.keras?sp=r&st=2026-01-02T09:40:54Z&se=2026-01-02T17:55:54Z&spr=https&sv=2024-11-04&sr=b&sig=5pYZKOVPLyqCV3AtdiVaRoPVoTVLqCROw%2BofTsJ32NI%3D"

model = None  # lazy-loaded singleton

# ---------------- DOWNLOAD MODEL ----------------
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists locally — skipping download")
        return

    print("📥 Downloading model from Azure Blob Storage...")
    try:
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print("✅ Model downloaded successfully")

    except Exception as e:
        print("❌ Model download failed:", e)
        raise RuntimeError("Model download failed")

# ---------------- LAZY LOAD MODEL ----------------
def get_model():
    global model

    if model is None:
        print("🧠 Loading model into memory (first request only)...")

        download_model()

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("✅ Model loaded into memory")

    return model

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    mdl = get_model()  # loads only once

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = load_img(filepath, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    prediction = mdl.predict(img_array, verbose=0)
    score = float(prediction[0][0])

    label = "Dog" if score >= 0.5 else "Cat"
    confidence = round(max(score, 1 - score) * 100, 2)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        image_path=filepath
    )

# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
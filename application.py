import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array

# --------------------------------------------------
# FORCE CPU (Azure App Service safe)
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (240, 240)
CLASS_NAMES = ["Cat", "Dog"]

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# Model paths
# --------------------------------------------------
MODEL_DIR = "/home/site/wwwroot/model"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model_savedmodel")
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔗 Azure Blob URL (FIXED MODEL)
MODEL_URL = "https://cnnmodelh5.blob.core.windows.net/cnnmodel/cnn_model_savedmodel.zip?sp=r&st=2026-01-02T19:54:19Z&se=2026-01-03T04:09:19Z&spr=https&sv=2024-11-04&sr=b&sig=elQi9NLktjRMh44WVDtq3XYpcZrGObt63q%2BjyLhoANw%3D"

# Singleton model (lazy loaded)
model = None

# --------------------------------------------------
# Download model if not present
# --------------------------------------------------
import zipfile

def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists — skipping download")
        return

    zip_path = os.path.join(MODEL_DIR, "model.zip")

    r = requests.get(MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    print("✅ SavedModel extracted")


# --------------------------------------------------
# Lazy model loader (loads only once)
# --------------------------------------------------
def get_model():
    global model

    if model is None:
        print("🧠 Loading model into memory...")
        download_model()

        model = tf.keras.models.load_model(MODEL_PATH)


        print("✅ Model loaded successfully")

    return model

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    mdl = get_model()

    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ---------------- Image preprocessing ----------------
    img = load_img(filepath, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Must match training preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # ---------------- Prediction ----------------
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

# --------------------------------------------------
# Local run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
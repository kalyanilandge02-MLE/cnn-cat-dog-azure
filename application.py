import os
import requests
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Config
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/cnn_model.h5"
IMG_SIZE = (224, 224)   # MUST match training size

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cnn_model.h5")

MODEL_URL = "https://cnnmodelh5.blob.core.windows.net/cnnmodel/cnn_model.h5"

os.makedirs("model", exist_ok=True)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Blob Storage...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully")

download_model()
model = load_model(MODEL_PATH)

CLASS_NAMES = ["Cat", "Dog"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=filepath
    )

# if __name__ == "__main__":
#     app.run()

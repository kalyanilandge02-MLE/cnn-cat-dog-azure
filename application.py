import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Config
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (240, 240)  # matches model input
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URL = "https://cnnmodelh5.blob.core.windows.net/cnnmodel/cnn_model.keras?sp=r&st=2026-01-01T18:18:06Z&se=2026-01-30T02:33:06Z&sv=2024-11-04&sr=b&sig=BxQExZck7jJ7wx0SqTOyUchMMg4k%2BulbZdgCMkFZKnQ%3D"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found locally.")
        print("🌐 Downloading model from Azure Blob Storage...")
        print("🔗 Source:", MODEL_URL)

        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model downloaded from Azure ({size_mb:.2f} MB)")
    else:
        print("✅ Model already exists locally — skipping Azure download")

# Download & load model
download_model()
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    prediction = model.predict(preprocessed_img)
    score = prediction[0][0]  # assumes output shape (1,1)
    
    # Compute class probabilities
    cat_prob = 1 - score
    dog_prob = score
    predicted_class = "Cat" if cat_prob > dog_prob else "Dog"
    confidence = round(max(cat_prob, dog_prob) * 100, 2)

    print(f"Predicted probability: {prediction} -> {predicted_class} ({confidence}%)")

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run()

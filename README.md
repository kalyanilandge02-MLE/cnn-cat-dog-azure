# CNN Cat and Dog Classification

This project is a Convolutional Neural Network (CNN) application for classifying images of cats and dogs. It uses Keras and TensorFlow for model development and Flask for serving a web interface.

## Features
- Trained CNN model for binary image classification (cat vs. dog)
- Web interface for uploading images and viewing predictions
- Organized project structure for easy extension and maintenance

## Project Structure
```
application.py                # Main Flask application
requirements.txt              # Python dependencies
model/
  cnn_model.keras             # Trained Keras model
  Cnn - cat and dog classification.ipynb  # Model training notebook
static/
  uploads/                   # Uploaded images
  data-.../                  # Data folders
templates/
  index.html                 # Web interface template
best-env/                     # Python virtual environment
```

## Setup Instructions
1. **Clone the repository**
2. **Create and activate a virtual environment** (or use the provided `best-env`)
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python application.py
   ```
5. **Open your browser** and go to `http://localhost:5000`

## Usage
- Upload an image of a cat or dog via the web interface.
- The app will predict and display whether the image is a cat or a dog.

## Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies

## License
This project is for educational purposes.

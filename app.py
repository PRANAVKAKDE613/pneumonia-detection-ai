from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "pneumonia_model_v2.keras"       # ← changed
IMG_SIZE = 224
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ── Download & Convert Model ──────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1ltiQdKghW1skFS0jubdYF_wz0jANt6KY"
    gdown.download(url, "pneumonia_model_fixed.h5", quiet=False)
    m = tf.keras.models.load_model("pneumonia_model_fixed.h5", compile=False)
    m.save(MODEL_PATH)
    print("Model converted and saved!")

# ── Load Model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        result = "PNEUMONIA"
        confidence = prediction * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction) * 100
    return result, round(float(confidence), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Invalid file type. Only PNG and JPG allowed."
        else:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            try:
                result, confidence = predict_image(filepath)
                image_path = filepath
            except ValueError as e:
                error = str(e)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
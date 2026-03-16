from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ─── Model Setup ───────────────────────────────────────────────────────────────
MODEL_PATH = "pneumonia_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1qa8at0d4AWOiI38g0yFTbRnmQEDOaZU_"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = "static/uploads"


# ─── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("Could not read the image file. Please upload a valid image.")

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


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        # Validate: file present
        if not file or file.filename == "":
            error = "No file selected. Please upload an X-ray image."

        # Validate: allowed extension
        elif not allowed_file(file.filename):
            error = "Invalid file type. Please upload a PNG or JPG image."

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
        error=error,
    )


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
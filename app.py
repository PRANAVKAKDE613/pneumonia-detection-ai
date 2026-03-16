from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown

app = Flask(__name__)

# ==============================
# MODEL DOWNLOAD FROM GOOGLE DRIVE
# ==============================

MODEL_PATH = "pneumonia_model.h5"

if not os.path.exists(MODEL_PATH):

    print("Downloading model from Google Drive...")

    url = "https://drive.google.com/uc?id=1qa8at0d4AWOiI38g0yFTbRnmQEDOaZU_"

    gdown.download(url, MODEL_PATH, quiet=False)


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

IMG_SIZE = 224


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_image(path):

    img = cv2.imread(path)
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

    return result, round(confidence, 2)


# ==============================
# MAIN ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        result, confidence = predict_image(filepath)

        image_path = filepath

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
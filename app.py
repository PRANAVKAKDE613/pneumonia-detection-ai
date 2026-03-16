from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

model = tf.keras.models.load_model("pneumonia_model.h5")

IMG_SIZE = 224


def predict_image(path):

    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.reshape(img,(1,IMG_SIZE,IMG_SIZE,3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "PNEUMONIA"
        confidence = prediction * 100
    else:
        result = "NORMAL"
        confidence = (1-prediction) * 100

    return result, round(confidence,2)


@app.route("/", methods=["GET","POST"])
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


if __name__ == "__main__":
    app.run(debug=True)
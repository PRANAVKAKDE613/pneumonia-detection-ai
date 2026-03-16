import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224

model = tf.keras.models.load_model("pneumonia_model.h5")

def predict_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.reshape(img,(1,IMG_SIZE,IMG_SIZE,3))

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print("PNEUMONIA DETECTED")
    else:
        print("NORMAL")

predict_image("dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
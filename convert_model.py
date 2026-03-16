import tensorflow as tf
import numpy as np

# Rebuild the exact same model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Load weights from old model
old_model = tf.keras.models.load_model("pneumonia_model_fixed.h5", compile=False)
model.set_weights(old_model.get_weights())

# Save in new format
model.save("pneumonia_model_v2.keras")
print("Done!")
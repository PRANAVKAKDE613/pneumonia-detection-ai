import tensorflow as tf
from tensorflow.keras.layers import InputLayer

class CompatInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        super().__init__(*args, **kwargs)

# Load old model
model = tf.keras.models.load_model(
    "pneumonia_model_fixed.h5",
    compile=False,
    custom_objects={"InputLayer": CompatInputLayer}
)

# Re-save in new format
model.save("pneumonia_model_v2.keras")
print("Done! Model saved as pneumonia_model_v2.keras")
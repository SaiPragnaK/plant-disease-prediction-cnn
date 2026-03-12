import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# class labels
classes = [
    "Apple Scab",
    "Black Rot",
    "Healthy"
]

# image path
img_path = "test_leaf.jpg"

# load image
img = image.load_img(img_path, target_size=(128,128))

# convert to array
img_array = image.img_to_array(img)

# normalize
img_array = img_array/255.0

# add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# prediction
prediction = model.predict(img_array)

print("Prediction:", classes[np.argmax(prediction)])
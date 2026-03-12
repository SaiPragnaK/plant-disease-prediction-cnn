from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Core Model Logic
model = tf.keras.models.load_model("plant_disease_model.h5")
classes = ["Apple Scab", "Black Rot", "Healthy"]

# Simplified Tips
TIPS = {
    "Apple Scab": [
        "Remove and clear away fallen leaves.",
        "Prune branches to let in more sunlight.",
        "Apply organic fungicide if the problem persists."
    ],
    "Black Rot": [
        "Cut off and dispose of infected areas.",
        "Keep the leaves dry when watering.",
        "Clean your garden tools after every use."
    ],
    "Healthy": [
        "Your plant looks great!",
        "Keep up with regular watering.",
        "Ensure it gets enough natural light."
    ]
}

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    current_tips = []

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join("static", filename)
            file.save(filepath)
            image_path = filename 

            img = image.load_img(filepath, target_size=(128,128))
            img_array = image.img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = model.predict(img_array)
            prediction = classes[np.argmax(pred)]
            confidence = float(np.max(pred) * 100)
            current_tips = TIPS.get(prediction, [])

    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path, tips=current_tips)

if __name__ == "__main__":
    app.run(debug=True)
import os
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load the trained model
model = tf.keras.models.load_model("tomato_leaf_disease_model.h5")

# Define class labels (same as dataset folders)
class_labels = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Function to process and predict image
def predict_disease(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize to model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get class index
    confidence = np.max(prediction)  # Get confidence score

    return class_labels[predicted_class], confidence

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            predicted_class, confidence = predict_disease(filepath)
            return render_template("index.html", uploaded_image=file.filename, prediction=predicted_class, confidence=confidence)

    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

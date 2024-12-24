import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

app = Flask(__name__)

# Load models
liver_model_path = "liver_model.pkl"
liver_model = joblib.load(liver_model_path)

# Load ONNX model for fatty liver
onnx_model_path = "fatty_liver_model.onnx"
onnx_session = ort.InferenceSession(onnx_model_path)

# Path to save images temporarily
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess the image to match ONNX model input shape
def preprocess_image(image):
    img = image.resize((227, 227))
    img = np.array(img).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to make predictions with the liver disease model
def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        result = liver_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template('home.html')

# Route for liver disease prediction
@app.route('/liver')
def liver():
    return render_template("liver.html")

@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
            if int(result) == 1:
                prediction = "Sorry, you have chances of getting the disease. Please consult the doctor immediately."
            else:
                prediction = "No need to fear! You have no dangerous symptoms of the disease."
            return render_template("result.html", prediction_text=prediction)

    return redirect(url_for('liver'))

# Route for ultrasound fatty liver prediction
@app.route('/ultrasound')
def ultrasound():
    return render_template('index.html')

@app.route('/predict_ultrasound', methods=['POST'])
def predict_ultrasound():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})

    if file:
        # Process the image without saving or displaying it
        image = Image.open(file.stream)

        # Preprocess image for model input
        input_image = preprocess_image(image)

        # Simulate a delay for prediction processing
        time.sleep(2)  # Simulate delay for model prediction
        inputs = {onnx_session.get_inputs()[0].name: input_image}
        output = onnx_session.run(None, inputs)
        prediction = np.argmax(output)

        # Map prediction to human-readable grade
        grade_map = {0: "Unknown", 1: "Grade 1", 2: "Grade 2", 3: "Grade 3"}
        predicted_grade = grade_map.get(prediction, "Normal")

        return jsonify({
            "status": "success",
            "prediction": predicted_grade
        })

    return jsonify({"status": "error", "message": "File processing failed"})



@app.route('/diet')
def diet_plan():
    return render_template('diet.html')

@app.route('/contact')
def contact_hospitals():
    return render_template('contact-hospitals.html')


if __name__ == "__main__":
    app.run(debug=True)

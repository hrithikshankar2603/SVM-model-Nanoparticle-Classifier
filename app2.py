from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load('best_model2.joblib')

# Define the target image size
target_size = (416, 416)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Read the image file and preprocess it
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()
    
    # Make a prediction using the trained SVM model
    prediction = model.predict([img])[0]
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

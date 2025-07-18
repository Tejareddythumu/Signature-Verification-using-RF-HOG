from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

app = Flask(__name__)

# Load your model and scaler
model = joblib.load("signature_verification_model_test.pkl")
scaler = joblib.load("scaler_test.pkl")


def extract_features_from_image(image_data):
    image = Image.open(image_data).convert('L')
    image = image.resize((128, 128))
    image = np.array(image)
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return features


@app.route('/')
def index():
    return send_from_directory('', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload two images'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    features1 = extract_features_from_image(image1)
    features2 = extract_features_from_image(image2)

    if features1 is None or features2 is None:
        return jsonify({'error': 'Error processing images'}), 500

    combined_features = np.concatenate((features1, features2)).reshape(1, -1)
    scaled_features = scaler.transform(combined_features)

    prediction = model.predict(scaled_features)[0]
    result = 'Genuine' if prediction == 1 else 'Forged'

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
import os
import sys

# This is a Flask App Made by demon developer
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

# Lets Import TensorFlow and tf.keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilities
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

MODEL_PATH = 'models/oldModel.h5'

# Loading the trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')


def model_predict(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # This is our Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # Save the image to a temporary file in Uploads Folder 
        img_path = os.path.join(os.path.dirname(__file__), 'uploads', 'image.jpg')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)  # Ensure the directory exists
        img.save(img_path)
        
        # Load the saved image in the Browser
        img = image.load_img(img_path, target_size=(64, 64))

        # Predict the class
        preds = model_predict(img, model)
        
        result = preds[0, 0]
        print(result)
        
        if result > 0.5:
            return jsonify(result="Patient is suffering with PNEUMONIA")
        else:
            return jsonify(result="Patient is NORMAL ❤️...")

    return None


if __name__ == '__main__':
    # Now Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()

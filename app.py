# app.py

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize to match model's expected sizing
    img = np.array(img) / 255.0  # Scale pixel values to [0, 1]
    img = img.reshape((1, 128, 128, 3))  # Reshape to match model's expected shape
    return img

# Image classification route
@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        # Check if the file is an image
        if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
            # Preprocess the image
            img = preprocess_image(file)
            
            # Perform classification
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            
            # Return the result to the frontend
            return render_template('index.html', prediction=predicted_class)
        else:
            return render_template('index.html', error='Invalid file type')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



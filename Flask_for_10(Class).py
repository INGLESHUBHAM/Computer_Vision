#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("imageclassifier_10.h5")

# Define the class labels
class_labels = ['Belly Dance', 'BharatNattyam', 'Flamenco', 'Jazz', 'Kathakali', 'Lion Dance', 'Waltz', 'Folklorico', 'Hiphop', 'samba']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Handle GET request (display a form, for example)
        return render_template('upload_form.html')  # You can create a new HTML file for the upload form
    elif request.method == 'POST':
        # Handle POST request (process the uploaded file and make predictions)
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            image = Image.open(file)
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_labels[predicted_class]
            return predicted_label

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





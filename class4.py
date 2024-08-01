#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the saved model
loaded_model = load_model('imageclassifier_04.h5')

# Image dimensions
img_width, img_height = 224, 224  # Set your desired image dimensions

# Class labels
class_labels = ['Jazz', 'Kathakali', 'Waltz', 'Folklorico']

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class_name = None

    if request.method == 'POST':
        # Get the uploaded image from the form
        uploaded_image = request.files['image']
        if uploaded_image:
            # Save the uploaded image to a temporary location
            img_path = 'static/uploaded_image.jpg'
            uploaded_image.save(img_path)

            # Load and preprocess the uploaded image
            img = image.load_img(img_path, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize the pixel values

            # Predict the class of the new image using the loaded model
            predicted_class = np.argmax(loaded_model.predict(img), axis=-1)
            predicted_class_name = class_labels[predicted_class[0]]

    return render_template('index.html', predicted_class_name=predicted_class_name)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





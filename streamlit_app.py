#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[1]:


import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# Load the trained model
model = keras.models.load_model(r"C:\Users\Ignitiv\Desktop\Multiclass\imageclassifier_06.h5")

# Define the class labels
class_labels = ['BharatNattyam','Jazz','Kathakali','Lion Dance','Waltz','Folklorico']

# Streamlit app
st.title("Image Classification Demo")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    # Display the prediction
    st.write(f"Prediction: {predicted_label}")


# In[4]:


pip install --upgrade protobuf


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





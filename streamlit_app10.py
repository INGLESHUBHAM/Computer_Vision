#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install --upgrade protobuf


# In[ ]:


# pip install streamlit


# In[ ]:


import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# Load the trained model
model = keras.models.load_model(r"C:\Users\Ignitiv\Desktop\Multiclass\imageclassifier_10.h5")

# Define the class labels
class_labels = ['Belly Dance', 'BharatNattyam', 'Flamenco', 'Jazz', 'Kathakali', 'Lion Dance', 'Waltz', 'Folklorico', 'Hiphop', 'samba']

# Streamlit app
st.title("Image Classification Demo")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "bmp", "tiff"])

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


# In[ ]:





# In[ ]:





# In[ ]:


# import os
# import numpy as np
# import streamlit as st
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# # Load the saved model
# loaded_model = load_model('imageclassifier_10.h5')

# # Image dimensions
# img_width, img_height = 224, 224  # Set your desired image dimensions

# def predict_image_class(image_path):
#     # Load the image for prediction
#     img = image.load_img(image_path, target_size=(img_width, img_height))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.0  # Normalize the pixel values

#     # Predict the class of the new image using the loaded model
#     predicted_class = np.argmax(loaded_model.predict(img), axis=-1)
#     class_labels = ['Belly Dance', 'BharatNattyam', 'Flamenco', 'Jazz', 'Kathakali', 'Lion Dance', 'Waltz', 'Folklorico', 'Hiphop', 'samba']
#     predicted_class_name = class_labels[predicted_class[0]]

#     return predicted_class_name

# def main():
#     st.title("Image Classifier Demo")

#     # Get user input for the image file path
#     new_image_path = st.text_input("Enter the path of the new image:")

#     # Check if the file exists
#     if not os.path.exists(new_image_path):
#         st.error(f"Error: The file {new_image_path} does not exist.")
#         return

#     # Check if the file is an image
#     allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
#     if not any(new_image_path.lower().endswith(ext) for ext in allowed_extensions):
#         st.error("Error: The provided file is not a valid image.")
#         return

#     # Predict the class and display the result
#     predicted_class_name = predict_image_class(new_image_path)

#     st.subheader(f'The predicted class for the new image is: {predicted_class_name}')

#     # Visualize the new image
#     st.image(new_image_path, caption=f'Predicted Class: {predicted_class_name}', use_column_width=True)

# if __name__ == "__main__":
#     main()


# In[ ]:





# In[ ]:





# In[ ]:





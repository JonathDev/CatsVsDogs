import streamlit as st
import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal

# Importer les modèles sauvegardés
model_paths = {
    "Model 1": "model1_save.h5",
    "Model 2": "model2_save.h5",
    "Model 3": "model3_save.h5",
    "Model 4": "model4_save.h5"
}

# Charger les modèles
models = {}
for name, path in model_paths.items():
    models[name] = tf.keras.models.load_model(path)

# Fonction pour faire une prédiction sur une image
def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(x)

# Streamlit app
st.title("Modèles de classification d'images")

# Sélection du modèle
model_name = st.sidebar.selectbox("Choisir un modèle", list(models.keys()))

# Option pour télécharger une image localement
uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])

# Option pour entrer un lien URL
url = st.text_input("Entrer l'URL de l'image")

# Option pour utiliser la webcam (à implémenter)
use_webcam = st.checkbox("Utiliser la webcam")

# Faire une prédiction lorsque l'utilisateur soumet une image
if st.button("Prédire"):
    if uploaded_file is not None:
        # Prédiction sur l'image téléchargée
        prediction = predict_image(models[model_name], uploaded_file)
        st.write(f"Prediction pour l'image téléchargée: {prediction}")
        # Afficher l'image téléchargée
        st.image(uploaded_file, caption='Image Téléchargée', use_column_width=True)
    elif url:
        # Prédiction sur l'image en ligne
        # Implémentez la fonction predict_image pour prendre une URL en entrée
        prediction = None  # Appel de la fonction pour prédire l'image en ligne
        st.write(f"Prediction pour l'image en ligne: {prediction}")
        # Afficher l'image en ligne
        st.image(url, caption='Image en Ligne', use_column_width=True)
    elif use_webcam:
        # Prédiction sur l'image de la webcam
        # Implémentez la fonction pour capturer l'image de la webcam et la prédiction
        st.warning("Fonctionnalité de la webcam non implémentée pour le moment.")
    else:
        st.error("Veuillez uploader une image, entrer une URL ou activer la webcam.")


import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from CatsVsDogs.utility import preprocess_data


train_generator, validation_generator = preprocess_data()


conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
conv_base.summary()

# Geler les poids du modèle VGG16
conv_base.trainable = False

# Créer le modèle
model4 = tf.keras.models.Sequential([
    conv_base, 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Pour une classification binaire
])
model4.summary()

model4.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics=['acc'])


history = model4.fit(
    train_generator, 
    steps_per_epoch=1247, # nombre d'images = batch_size * steps
    epochs=50,  # Utilisez "epochs" au lieu de "epoch" pour spécifier le nombre d'époques
    validation_data=validation_generator, 
    validation_steps=65  # nombre d'images = batch_size * steps
)

model4.save('mode4_save.h5')
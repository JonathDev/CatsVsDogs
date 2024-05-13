import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from CatsVsDogs.utility import preprocess_data

train_generator, validation_generator = preprocess_data()

model3 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape = (150,150,3), activation='relu'), # couche flatten 
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(), # couche flatten 
    tf.keras.layers.Dropout(0.5),# il va supprimmer 50% de mes poids  permet de combattre l'overfitting
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid') # pour un probleme de classification binaire  on met un seul neurone, elle va retourner entre 0 et 1
])
model3.summary()

model3.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics=['acc'])

history = model3.fit(
    train_generator, 
    steps_per_epoch=1247, # nombre d'images = batch_size * steps
    epochs=50,  # Utilisez "epochs" au lieu de "epoch" pour spécifier le nombre d'époques
    validation_data=validation_generator, 
    validation_steps=65  # nombre d'images = batch_size * steps
)

model3.save('mode3_save.h5')
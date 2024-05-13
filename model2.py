import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from CatsVsDogs.utility import preprocess_data

train_generator, validation_generator = preprocess_data()


model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape = (150,150,3), activation='relu'), # couche flatten 
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(), # couche flatten 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid') # pour un probleme de classification binaire  on met un seul neurone, elle va retourner entre 0 et 1
])
model2.summary()

model2.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics=['acc'])

history = model2.fit(
    train_generator, 
    steps_per_epoch=1247, # nombre d'images = batch_size * steps
    epochs=50,  # Utilisez "epochs" au lieu de "epoch" pour spécifier le nombre d'époques
    validation_data=validation_generator, 
    validation_steps=44  # nombre d'images = batch_size * steps
)

model2.save('model2_save.h5')

data_img = '/home/jonat/Documents/Formation/Cours 4 Deep Learning/Cours udemy/Projet chien et chat/image/722x460_berger-malinois-illustration.webp'
img = image.load_img(data_img , target_size=(150,150))
x= image.img_to_array(img)
x = x[np.newaxis]
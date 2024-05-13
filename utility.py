import os
import tensorflow as tf

def preprocess_data():
    # Chemin relatif par rapport au répertoire de travail actuel
    base_dir = os.getcwd()  # Obtient le répertoire de travail actuel
    
    # Définition des répertoires
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "test")

    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir =os.path.join(validation_dir, 'dogs')

    # Vérification si les répertoires existent
    if not os.path.exists(train_cats_dir):
        print("Le répertoire train_cats_dir n'existe pas:", train_cats_dir)
    else:
        train_cat_fnames = os.listdir(train_cats_dir)
        train_dog_fnames = os.listdir(train_dogs_dir)
        validation_cat_fnames = os.listdir(validation_cats_dir)
        validation_dog_fnames =os.listdir(validation_dogs_dir)
        print(train_cat_fnames[:11])
        print(train_dog_fnames[:12])
        print(validation_cat_fnames[:11])
        print(validation_dog_fnames[:12])
        


    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.0, 
        rotation_range=40,  
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        shear_range=0.2,  
        zoom_range=0.2,  
        horizontal_flip=True,  
        fill_mode='nearest'
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(150,150), 
        batch_size=20,
        class_mode='binary'
    )

    validation_generator =  test_datagen.flow_from_directory(
        validation_dir, 
        target_size=(150,150), 
        batch_size=20,
        class_mode='binary'
    )

    return train_generator, validation_generator
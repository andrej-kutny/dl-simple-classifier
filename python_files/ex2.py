import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

NUMBER_OF_EPOCHS = 1
LEARNING_RATE = 0.0001

epochs = NUMBER_OF_EPOCHS

image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "lab1/cat_and_dog_images",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

loaded_model = keras.saving.load_model("lab1/models/lab1b_40_epochs.keras")

loaded_model.layers[-1] = layers.Dense(1, activation=None)(loaded_model.layers[-2].output)

callbacks = [
    keras.callbacks.ModelCheckpoint("ex2_save_at_{epoch}.keras"),
]
loaded_model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

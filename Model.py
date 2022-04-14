import DataLoader
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

ImgSize = (64, 64)

print("Loading Data\n")
TrainDs, ValDs, Classes = DataLoader.LoadData(ImgSize)

Model = Sequential([
    layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(ImgSize[0], ImgSize[1], 3)),

    layers.Conv2D(32, 4, padding='same', activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(64, 4, padding='same', activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),
    layers.Flatten(),

    layers.Dense(len(Classes), activation='sigmoid')
])

Model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

print("\nSummary of Model\n")
Model.summary()


epochs = 10
print(f"\nTraining model for {epochs} Epochs\n")

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=160)

History = Model.fit(
    TrainDs,
    validation_data=ValDs,
    epochs=epochs,
)

Model.save("TrainedModel")

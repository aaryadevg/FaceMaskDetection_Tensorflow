import DataLoader
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

ImgSize = (64, 64)

print("Loading Data\n")
TrainDs, ValDs, Classes = DataLoader.LoadData(ImgSize)

Model = tf.keras.models.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Rescaling(
            1.0 / 255, input_shape=(ImgSize[0], ImgSize[1], 3)
        ),
        tf.keras.layers.Conv2D(16, (2,2), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, (2,2), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

Model.compile(optimizer='adam',
              loss= tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

print("\nSummary of Model\n")
Model.summary()


epochs = 10
print(f"\nTraining model for {epochs} Epochs\n")

checkpoint_path = "training"
os.makedirs(checkpoint_path, exist_ok=True)
checkpoint_dir = os.path.join(checkpoint_path, "cp-{epoch:04d}.ckpt")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq="epoch")

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=2
)

History = Model.fit(
    TrainDs,
    validation_data=ValDs,
    epochs=epochs,
    verbose = 2,
    callbacks=[cp_callback, es_callback] 
)

Model.save("TrainedModel")

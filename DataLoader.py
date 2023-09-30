import os
import tensorflow as tf


def LoadData(ImgSize, BatchSize=32):
    """
    LoadData Loads data from disk and converts to a Tensorflow Dataset

    Arguments:
        ImgSize {tuple(int)} -- Size of the input images (W,H) 
        BatchSize {int} -- Number of images in 1 batch of the dataset (Defaults to 32)

    Returns:
        Tensorflow Dataset containing training and testing data and number classes in the dataset
    """
    TRAIN_DIR = "Face Mask Dataset/Train"
    TEST_DIR = "Face Mask Dataset/Test"

    TrainingDS = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=ImgSize,
        batch_size=BatchSize)

    ValidationDS = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        subset="validation",
        seed=123,
        image_size=ImgSize,
        batch_size=BatchSize,
        shuffle=False)

    ClassNames = os.listdir(TRAIN_DIR)

    # Caching tensorflow dataset objects
    TrainingDS.cache("Cache/Training.Data")
    ValidationDS.cache("Cache/Validation.Data")

    return (TrainingDS, ValidationDS, ClassNames)


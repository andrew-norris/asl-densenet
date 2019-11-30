import os
import glob
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage

DATA_DIR = os.path.abspath('~/.keras/datasets')

DATA_URL = "http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2"


def download_dataset(dataset_download_path):

    tf.keras.utils.get_file(origin=DATA_URL, fname='asl_fingerspelling', extract=True)

    depth_files_path = os.defpath.join(dataset_download_path, '**/depth*')

    file_list = glob.glob(depth_files_path, recursive=True)

    for file_path in file_list:
        try:
            os.remove(file_path)
        except OSError:
            print("error")

    train_datagen = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    dataset_path = os.path.join(dataset_download_path, 'A/')

    train_generator = train_datagen.flow_from_directory(
        directory=r""+dataset_path,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    valid_dataset_path = os.path.join(dataset_download_path, 'D/')

    valid_generator = valid_datagen.flow_from_directory(
        directory=r"" + valid_dataset_path,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    return train_generator, valid_generator


def get_next_generator(dataset_download_path, directory):
    dataset_path = os.path.join(dataset_download_path, directory)

    train_datagen = ImageDataGenerator()
    return train_datagen.flow_from_directory(
        directory=r""+dataset_path,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

def download_pretrained_weights():
    client = storage.Client(project='asl-densenet')
    bucket = client.get_bucket('asl-densenet-final-project')
    weights = bucket.get_blob('imagenet_models/densenet161_weights_tf.h5')

    wd = os.getcwd()
    print(wd)

    weights.download_to_filename('/root/densenet161_weights_tf.h5', client)
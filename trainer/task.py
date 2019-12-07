import trainer.utils as utils
from trainer.model import dense_net
import argparse
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def get_args():
  """Argument parser.

  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='local or GCS location for writing checkpoints and exporting models')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=4,
      help='number of times to go through the data, default=20')
  parser.add_argument(
      '--batch-size',
      default=32,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning-rate',
      default=0.001,
      type=float,
      help='learning rate for gradient descent, default=.01')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')
  parser.add_argument(
      '--decay',
      default=0.0001,
      type=int)
  parser.add_argument(
      '--optimizer',
      default=0,
      type=int)
  parser.add_argument(
      '--dataset-size',
      default=1,
      type=int
  )

  args, _ = parser.parse_known_args()
  return args


WEIGHTS_PATH = '/root/densenet161_weights_tf.h5'
IS_LOCAL = False
MODEL_EXPORT_PATH = '/root/keras_export.h5'
KERAS_PATH = '/root/.keras/'
DATASET_DOWNLOAD_PATH = os.path.join(KERAS_PATH, 'datasets/dataset5/')

def train(args):

    num_classes = 24
    batch_size = 32
    epochs = 1
    learning_rate = 0.001
    decay = 0.0001
    optimizer = 0
    set_size = 1


    utils.download_pretrained_weights()

    train_generator, valid_generator = utils.download_dataset(DATASET_DOWNLOAD_PATH)

    model = dense_net(
        num_classes=num_classes,
        learning_rate=learning_rate,
        decay=decay,
        optimizer=optimizer,
        weights_path=WEIGHTS_PATH
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=batch_size,
        validation_data=valid_generator,
        validation_steps=batch_size,
        epochs=epochs
    )

    directories = []
    if set_size == 3:
        directories = ['B/', 'C/']
    elif set_size == 4:
        directories = ['B/', 'C/', 'E/']

    if set_size != 1:
        for directory in directories:
            train_generator = utils.get_next_generator(DATASET_DOWNLOAD_PATH, directory)
            model.fit_generator(
                train_generator,
                steps_per_epoch=batch_size,
                validation_data=valid_generator,
                validation_steps=batch_size,
                epochs=epochs
            )

    model.evaluate_generator(
        valid_generator,
        steps=batch_size
    )

    if IS_LOCAL:
        export_path = MODEL_EXPORT_PATH
    else:
        export_path = os.path.join(args.job_dir, 'keras_export.h5')

    model.save(export_path)

    Y_pred = model.predict_generator(valid_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    print("Confusion Matrix")
    print(confusion_matrix(valid_generator.classes, y_pred))

    print("classification report")
    print(classification_report(valid_generator.classes, y_pred))

    print('Model exported to: {}'.format(export_path))

if __name__ == '__main__':
    args = get_args()
    train(args)
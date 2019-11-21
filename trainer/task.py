import trainer.utils as utils
from trainer.model import dense_net
import argparse
import tensorflow as tf
import os
from PIL import Image

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

def train(args):

    train_generator, valid_generator = utils.download_dataset()

    num_classes = 24
    batch_size = args.batch_size
    epochs = args.num_epochs
    learning_rate = args.learning_rate
    decay = args.decay
    optimizer = args.optimizer
    set_size = args.set_size

    model = dense_net(
        classes=num_classes,
        learning_rate=learning_rate,
        decay=decay,
        optimizer=optimizer
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
        directories = ['B/', 'C/', 'E']

    if set_size != 1:
        for directory in directories:
            train_generator = utils.get_next_generator(directory)
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

    export_path = os.path.join(args.job_dir, 'keras_export')
    model.save(export_path)
    print('Model exported to: {}'.format(export_path))

if __name__ == '__main__':
    args = get_args()
    train(args)
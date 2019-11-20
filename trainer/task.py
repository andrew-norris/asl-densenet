import trainer.utils as utils
from trainer.model import dense_net
import argparse
import tensorflow as tf
import os

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
      default=20,
      help='number of times to go through the data, default=20')
  parser.add_argument(
      '--batch-size',
      default=128,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning-rate',
      default=.01,
      type=float,
      help='learning rate for gradient descent, default=.01')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')
  args, _ = parser.parse_known_args()
  return args

def train(args):

    train_generator, valid_generator = utils.download_dataset()

    num_classes = 24
    batch_size = 32
    epochs = 2

    model = dense_net(classes=num_classes)

    model.fit_generator(
        train_generator,
        steps_per_epoch=batch_size,
        valid_generator=valid_generator,
        validation_steps=batch_size,
        epochs=epochs
    )

    # directories = ['B/', 'C/']
    # for directory in directories:
    #     train_generator = utils.get_next_generator(directory)
    #     model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=batch_size,
    #         valid_generator=valid_generator,
    #         validation_steps=batch_size,
    #         epochs=epochs
    #     )

    model.evaluate_generator(
        valid_generator=valid_generator,
        steps=batch_size
    )

    export_path = os.path.join(args.job_dir, 'keras_export')
    model.save(export_path)
    print('Model exported to: {}'.format(export_path))

if __name__ == '__main__':
    args = get_args()
    train(args)
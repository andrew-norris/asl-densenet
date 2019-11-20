import trainer.utils as utils
from trainer.model import dense_net

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

    directories = ['B/', 'C/']
    for directory in directories:
        train_generator = utils.get_next_generator(directory)
        model.fit_generator(
            train_generator,
            steps_per_epoch=batch_size,
            valid_generator=valid_generator,
            validation_steps=batch_size,
            epochs=epochs
        )

    model.evaluate_generator(
        valid_generator=valid_generator,
        steps=batch_size
    )
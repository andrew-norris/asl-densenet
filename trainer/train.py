# -*- coding: utf-8 -*-

from sklearn.metrics import log_loss
from trainer.loadData import load_data
from trainer.densenet161 import DenseNet
from keras.optimizers import SGD


def train():

    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_classes = 26
    batch_size = 16
    nb_epoch = 10



    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_data(img_rows, img_cols)

    # Load our model
    model = DenseNet(classes=num_classes)

    print('loaded model')

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )


    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print(score)


if __name__ == "__main__":
    train()
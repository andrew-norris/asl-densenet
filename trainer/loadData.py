import cv2
import numpy as np
from google.cloud import storage
import pathlib

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf
import glob

import os

'''
img_rows, img_cols = 224
'''
# export this function for getting training set and valid set
'''
img_rows, img_cols = 224
'''
# export this function for getting training set and valid set
def load_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    X_train, Y_train, X_valid, Y_valid = load_image()

    print('can we load images?')


    # Resize trainging images
    if K.image_data_format() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train])

    print('loaded_data')

    return X_train, Y_train, X_valid, Y_valid


def load_image():
    datasetFolder = 'trainer/dataset'
    alphabetList = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y'] # 24 letters

    signersListTrain = ['A','B','C','D'] # ['A','B','C','D']
    XTrains = []
    YTrains = []
    signers = np.array([item for item in glob.glob('/root/.keras/datasets/dataset5/*') if item != '/root/.keras/datasets/dataset5/E'])
    print(signers)
    for signer in signers:
        letters = np.array([item for item in glob.glob(signer + '/*')])
        for letter in letters:
            letter_dir = np.array([item for item in glob.glob(letter + '/*')])
            for file_name in letter_dir:
              img = cv2.imread(file_name)
              if img is not None:
                  XTrains.append(img)
                  YTrain = np.zeros(24)  # There are only 24 letters, excluding 'j' and 'z'
                  YTrain[alphabetList.index(letter[-1])] = 1
                  YTrains.append(YTrain)

    print('part 1 worked')
    # signersListValid = ['E']
    XValids = []
    YValids = []

    letters = np.array([item for item in glob.glob('/root/.keras/datasets/dataset5/E/*')])
    for letter in letters:
        letter_dir = np.array([item for item in glob.glob(letter + '/*')])
        for file_name in letter_dir:
            img = cv2.imread(file_name)
            if img is not None:
                XValids.append(img)
                YValid = np.zeros(24)  # There are only 24 letters, excluding 'j' and 'z'
                YValid[alphabetList.index(letter[-1])] = 1
                YValids.append(YValid)
    print('part 2 worked')
    return XTrains, YTrains, XValids, YValids
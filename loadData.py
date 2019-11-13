import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

import os

'''
img_rows, img_cols = 224
'''
# export this function for getting training set and valid set
def load_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid,Y_valid)= load_image()

    # Resize trainging images
    if K.image_data_format() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train])
        # X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:len(X_valid),:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train])
        # X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:len(X_valid),:,:,:]])

    return (X_train, Y_train), (X_valid, Y_valid)


def load_image():
    datasetFolder = 'dataset'
    alphabetList = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y'] # 24 letters

    signersListTrain = ['A','B','C','D'] # ['A','B','C','D']
    XTrains = []
    YTrains = []
    for signer in signersListTrain:
        for alphabet in [folder for folder in os.listdir(os.path.join(datasetFolder,signer)) if not folder.startswith('.')]:
            for filename in os.listdir(os.path.join(datasetFolder,signer,alphabet)):
                img = cv2.imread(os.path.join(datasetFolder,signer,alphabet,filename))
                if img is not None:
                    XTrains.append(img)
                    YTrain = np.zeros(24)  # There are only 24 letters, excluding 'j' and 'z'
                    YTrain[alphabetList.index(alphabet)] = 1
                    YTrains.append(YTrain)

    signersListValid = ['E']
    XValids = []
    YValids = []

    for signer in signersListValid:
        for alphabet in [folder for folder in os.listdir(os.path.join(datasetFolder,signer)) if not folder.startswith('.')]:
            for filename in os.listdir(os.path.join(datasetFolder,signer,alphabet)):
                img = cv2.imread(os.path.join(datasetFolder,signer,alphabet,filename))
                if img is not None:
                    XValids.append(img)
                    YValid = np.zeros(24)  # There are only 24 letters, excluding 'j' and 'z'
                    YValid[alphabetList.index(alphabet)] = 1
                    YValids.append(YValid)

    return (XTrains,YTrains),(XValids,YValids)


# (X_train, Y_train), (X_valid,Y_valid) = loadData(224,224)
#
# print('len(X_train)',len(X_train))
# print('len(Y_train)',len(Y_train))
# print('X_train[0]',X_train[0])
# print('Y_train[0]',Y_train[0])
# print('len X col row:',len(X_train[0]),len(X_train[0][0]))
# print('len Y col', len(Y_train[0]))

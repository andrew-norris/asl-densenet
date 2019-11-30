# asl-densenet: https://github.com/andrew-norris/asl-densenet

# Problem Definition
Individuals with a hearing impairment often find it challenging to communicate with those without hearing impairments. This can be a significant issue when those individuals need to convey important information such as appointments, when an interpreter is not available to assist.
We have decided to look into identifying American Sign Language signals to ‘help’ with this problem. To enhance our models classifications we want to combine the sign language signals with lip reading comprehension. We believe this will improve upon the classifications that are produced when only examining a single aspect of speech comprehension for people with hearing impairments.

# Densenet Model
For our model we used densenet161.py from: https://github.com/flyyufelix/DenseNet-Keras
This comes with pretrained weights that can be found at: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA

# Dataset
The dataset used to train our model can be found at: https://www.kaggle.com/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out
And downloaded with the following link: http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2

# Running the Code
In order to train the model locally you must first download the dataset and pretrained weights.
Then edit the variables:
    - IS_LOCAL: if the model is being run locally then set this to true
    - MODEL_EXPORT_PATH: Change this to edit where the finished model will be saved.
    - WEIGHTS_PATH: Change this to the path to the prerained weights that were downloaded.
    - KERAS_PATH: Change this to the local keras installation folder where the dataset will be downloaded to.

# Predicting
In order to predict you can use the demo colab that was set up for this purpose: https://colab.research.google.com/drive/1V9t259zVDHHqOM-hq865Vgwv_j59tjLJ
You will need to replace the downloaded weights with the keras_export.h5 weights that can be found in the project submission.
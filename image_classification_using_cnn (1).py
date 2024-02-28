# -*- coding: utf-8 -*-
"""Image classification using CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_v1ihM66TTbO9CHwRAMkNIAaJaw7Ce6K

Image Classification using CNN
"""

import numpy as np
import random
from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

"""**Importing datasets of dogs and cats**"""



from google.colab import drive
drive.mount('/content/drive')

X_train = np.loadtxt('/content/drive/MyDrive/Google Colab/input.csv', delimiter = ',')

X_test = np.loadtxt('/content/drive/MyDrive/Google Colab/input_test.csv', delimiter = ',')

Y_train = np.loadtxt('/content/drive/MyDrive/Google Colab/labels.csv', delimiter = ',')

Y_test = np.loadtxt('/content/drive/MyDrive/Google Colab/labels_test.csv', delimiter = ',')

"""Reshaping the numpy array to 100x100x3 matrix"""

X_train = X_train.reshape(len(X_train), 100,100,3)
X_test= X_test.reshape(len(X_test),100,100,3)

Y_train = Y_train.reshape(len(Y_train), 1)
Y_test= Y_test.reshape(len(Y_test),1)

import matplotlib.pyplot as plt

rd= random.randint(0, len(X_train))
plt.imshow(X_train[rd,:])
plt.show()

"""Creating model"""

model= Sequential([
      Conv2D(32, (3,3), activation ='relu', input_shape =(100,100,3)),
      MaxPooling2D((2,2)),

      Conv2D(32, (3,3), activation ='relu'),
      MaxPooling2D((2,2)),

      Flatten(),
      Dense(64, activation ='relu'),
      Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy' ,optimizer= 'adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size= 64)

model.evaluate(X_test, Y_test)

image_random= random.randint(0, len(Y_train))

plt.imshow(X_train[image_random, :])
plt.show()

y_pred= model.predict(X_train[image_random,:].reshape(1,100,100,3))
y_pred = y_pred > 0.5

if(y_pred==0):
  print("The image contains dog")
else:
  print("The image contains cat")


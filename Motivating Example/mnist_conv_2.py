from __future__ import print_function
import sys
import os
import numpy as np
import keras 
from keras import layers
import tensorflow as tf
import time 


num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


currentPATH= os.getcwd()

current = '{0}/{1}/'.format(currentPATH,'mnist_conv_2')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(layers.Conv2D(32,input_shape= (28, 28, 1), kernel_size=(3, 3), activation="relu") )
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))

batch_size = 128
epochs = 50
start_time = time.clock()
model.compile(loss="mean_absolute_error", optimizer=keras.optimizers.SGD(learning_rate = 0.001), metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[ tf.keras.callbacks.Metrics(x_train, y_train, len(model.layers), batch_size=batch_size, lr=0.001, PATH=current)])
print("Time:", str(time.clock() - start_time))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 
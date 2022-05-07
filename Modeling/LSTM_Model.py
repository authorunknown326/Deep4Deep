import tensorflow as tf 

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten, Activation, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import time 
import sys
import keras
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow import keras;    from tensorflow.keras import layers;
from tensorflow.keras import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from collections import Counter

from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN , SMOTETomek
#CondensedNearestNeighbour, AllKNN , InstanceHardnessThreshold, TomekLinks, OneSidedSelection


####### Handle balance data by using Oversampling 
sm = SMOTEENN()#k_neighbors = 1
X_resampled, y_resampled = sm.fit_resample(listArray, yt)
print(X_resampled.shape)
x_shape, y_shape = X_resampled.shape
print(y_resampled.shape)
counter = Counter(y_resampled)
print(counter)
label_sample =[]
for ysample in y_resampled:
    label_sample.append(inverse_labels[ysample])
    

X_resampled = X_resampled.reshape(x_shape, 47,683)

#print("[INFO] class labels:")
#mlb = MultiLabelBinarizer()
#
#labels = mlb.fit_transform(label_sample)
from keras.utils.np_utils import to_categorical
labels = to_categorical(label_sample)
#if np.isnan(arr.max()) or np.isnan(arr.min()):
#    print("csv")

X_shuffled,y_shuffled = shuffle(X_resampled, labels, random_state=42) #, random_state=0
x_train, x_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled,  test_size=0.30, shuffle=False)

print(y_test)

################# Model Arch

model_lstm = tf.keras.Sequential()

model_lstm.add(tf.keras.layers.BatchNormalization(input_shape=(None, 683)))
model_lstm.add(    tf.keras.layers.LSTM(683,  return_sequences =True,    dtype=tf.float32))#input_shape=(None, 146),
model_lstm.add(     tf.keras.layers.LSTM(683, return_sequences =False))
model_lstm.add(tf.keras.layers.Dropout(rate =0.2))

model_lstm.add(     tf.keras.layers.Flatten())

model_lstm.add(      tf.keras.layers.RepeatVector(2))
model_lstm.add(     tf.keras.layers.LSTM(683, return_sequences =True))
model_lstm.add(      tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

model_lstm.add(     tf.keras.layers.LSTM(683, return_sequences =True))

model_lstm.add(      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=5, activation = 'softmax')))


################# Model Training 
model_lstm.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=[keras.metrics.Precision(), keras.metrics.Recall(),keras.metrics.BinaryAccuracy() ])#keras.metrics.Precision(), keras.metrics.Recall()
history = model_lstm.fit(x_train,y_train, validation_split = 0.3,epochs = 15, batch_size = 16 )#, batch_size = 16, 
model_lstm.save('Classifier_Encoder_Decoder.h5')
model_lstm.evaluate(x_test, y_test, verbose=1)



################# Model Predict 

y_pred = model_lstm.predict(listArrayTest)

for x in y_pred:
    print(labelA(np.argmax(x[0])), labelA(np.argmax(x[1])))
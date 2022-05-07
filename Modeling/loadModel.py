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
scalar = MinMaxScaler()

def labelA(s):
    if s == 0:
        return 'Loss'
    if s == 1:
        return 'Batch_Size' 
    if s == 2:
        return 'act'
    if s == 3:
        return 'opt'
    if s == 4:
        return 'lr'
    if s == 5:
        return 'Dropout'
    if s == 6:
        return 'weights'
    if s == 7:
        return 'correct'
        
    
modelTestList = []


DI = 2
x_t = genfromtxt(r"PATH/A MOTIVATING EXAMPLE/DI/{0}/merge/Metrics.csv".format(DI), delimiter=',')
scalar.fit(x_t)
x_t = scalar.transform(x_t)
print(np.array(modelTestList).shape)
modelTestList.append(x_t)

         

model_lstm = keras.models.load_model("Classifier_Encoder_Decoder1.h5")


listArrayTest = np.array(modelTestList)

y_pred = model_lstm.predict(listArrayTest)

for x in y_pred:
    print(labelA(np.argmax(x[0])), labelA(np.argmax(x[1])))
from __future__ import print_function

import keras, sys

import os

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

import tensorflow as tf

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# here we will import the libraries used for machine learning

import numpy as np # linear algebra

import pandas as pd # data processing, data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph



from sklearn.linear_model import LogisticRegression # Logistic regression model

from sklearn.model_selection import train_test_split # to split the data into training and test set

from sklearn.model_selection import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter of models

from sklearn.ensemble import RandomForestClassifier # for random forest classifier model

from sklearn.neighbors import KNeighborsClassifier # for K Neighbors model

from sklearn.tree import DecisionTreeClassifier # for Decision Tree model

from sklearn import svm # for Support Vector Machine model

from sklearn import metrics # for the check the error and accuracy of the model

from sklearn.preprocessing import StandardScaler # To norm the data

    

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout









data=pd.read_csv('data.csv') # import from a csv file

data.drop("Unnamed: 32",axis=1,inplace=True) # delete unnecessary columns



#### transform the problem into binary classification : Malignant = 1 ans Benign = 0 ###

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0}) 

data.head(10) # show data





# As I said above the data can be divided into three parts corresponding to 3 dimensional features '(mean, se, worst)' 

# computed with the  the 3 dimensonial values (X,Y,Z)



features_mean= list(data.columns[2:12]) # mean group

features_se= list(data.columns[12:22]) # standard error group

features_worst=list(data.columns[22:32]) # features_worst group



print("-----------------------------------")

print('Mean set of all features')

print(features_mean)

print(len(features_mean), 'features')

print("-----------------------------------")

print('Standard Error set of all features')

print(features_se)

print(len(features_se), 'features')

print("------------------------------------")

print('Worst set of all features')

print(features_worst)

print(len(features_worst), 'features')

print(' ')

print('Description of data columns')

data.iloc[:,1:].describe() # description of all columns in the dataset (30 features + diagnosis)





#####  delete features higly corrolated described above  ###########



col_to_drop_corrolated1=['radius_mean','radius_se','radius_worst','area_mean','area_se','area_worst']

col_to_drop_corrolated2=['concavity_mean','concave points_mean', 'concavity_se','concave points_se' ,'concavity_worst','concave points_worst']

data.drop(col_to_drop_corrolated1+col_to_drop_corrolated2,axis=1,inplace=True)

print('I keep only ', len(data.columns) ,' features which are not so corrolated based on the previous analysis')



col_to_drop_mean=['fractal_dimension_mean', 'symmetry_mean', 'smoothness_mean', 'texture_mean'] # mean variables not efficient for detection

col_to_drop_se=['fractal_dimension_se', 'symmetry_se', 'smoothness_se', 'texture_se'] # Standard error variables not efficient for detection

col_to_drop_worst=['fractal_dimension_worst'] # Worst variable not efficient for detection

col_to_drop_tot=col_to_drop_mean+col_to_drop_se+col_to_drop_worst

data.drop(col_to_drop_tot,axis=1,inplace=True)

print('Now the data set is only composed of ', len(data.columns), 'features')



color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

#pd.plotting.scatter_matrix(data.iloc[:,2:], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix





prediction_var=data.columns[2:]

outcome_var= "diagnosis"



currentPATH= os.getcwd()



current = '{0}/{1}/'.format(currentPATH,'Metrics_53')



 #Fit the model:

train, test = train_test_split(data, test_size = 0.3) # in this our main data is splitted into train (70%) and test (30%) into the function (local variables)

train_X = train[prediction_var] # taking the training data input 

train_y=train.diagnosis # This is output of our training data



# same for data test

test_X= test[prediction_var] # taking test data inputs

test_y =test.diagnosis   #output value of test data



# norm the data with mean of 0 and standard deviation of 1

sc = StandardScaler() # sklearn object

sc.fit_transform(data[prediction_var]) 

train_X = sc.transform(train_X) #transform train set with the scaler method

test_X = sc.transform(test_X) # transform test set with scaler methodls





model = Sequential()
model.add(Dense(16 , kernel_initializer="random_normal",  input_dim=9))
model.add(Activation('relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001), loss='kullback_leibler_divergence', metrics=['accuracy'])
model.fit(train_X, train_y, batch_size=75, epochs=120, callbacks=[tf.keras.callbacks.Metrics(train_X, train_y, len(model.layers), batch_size=75, lr=0.001, PATH=current)  ])

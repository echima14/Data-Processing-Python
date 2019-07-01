# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:28:42 2017

@author: echima
"""

#Importing Libraries
import numpy as np # contains math tools
import matplotlib.pyplot as plt # helps with plotting charts
import pandas as pd # importing and managing datasets

#import the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
                
                
#Correct for missing data
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_x.fit_transform(y)

 #Split into test set and training set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test) 

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:29:19 2017

@author: echima
"""

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
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values # Matrix of independent variables
y = dataset.iloc[:,1].values # Matrix for depedent variables
                
 #Split into test set and training set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)


#Fitting simple linear regression to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict test results 
y_pred = regressor.predict(x_test)

# Visualzing the results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
plt.savefig('trainingSet.png')


# Visualizing the test results

plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
plt.savefig('slrTest.png')

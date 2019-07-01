

#importing libraries
import numpy as np
import matplotlib as plt
import pandas as pd

#import datasets
dataset = pd.read_csv('StartUp.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values
                
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid the Dummy variable trap
X = X[:,1:]

#Splitting the data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Fit the Multi Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

#Build the best model using backwards elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)

X_optimal = X[:,[0,1,2,3,4,5]]#creates a new matrix of features for us
#New stats model (New Regressor)
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()

regressor_OLS.summary()
#Seconnd Pass
X_optimal = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_OLS.summary()
#Third PAss
X_optimal = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_OLS.summary()

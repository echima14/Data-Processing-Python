# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import dataset
dataset = pd.read_csv('Position_Salary_Data_Set.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
            
# Fitting the regressor to the dataset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,Y)

#Fitting the Polynomial Regressor to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 2)
X_Polynomial = polyReg.fit(X)
linReg_2 = LinearRegression()
linReg_2.fit(X_Polynomial,Y)

# Predicting the results
predictions = (linReg.predict(6.5))

# Visualizations for Linear Regression
plt.scatter(X,Y, color  = 'red')
plt.plot(X,linReg.predict(X), color = 'blue')
plt.title('Salary Truth Detector(Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression
plt.scatter(X,Y,color = 'red')
plt.plot(X, linReg_2.predict(polyReg.fit_transform(X)), color = 'blue') 
plt.title('Salary Truth Detector(Polynomial Regressor Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

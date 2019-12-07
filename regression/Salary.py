# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 06:33:35 2019

@author: vinay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

#Divide dataset into X and Y
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting into train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)


#implementing simple linear regression
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(X_train,Y_train)
y_Predict = simpleLinearRegression.predict(X_test)

#showing linear regression
plt.scatter(X_train, Y_train,color = 'red')
plt.plot(X_train,simpleLinearRegression.predict(X_train))
plt.show()

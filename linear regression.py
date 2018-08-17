# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:58:27 2018

@author: user
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
import matplotlib.pyplot as mpp
import seaborn as sns
'''
#LINEAR REGRESSION - 
def lr(x,y):
    model=linear_model.LinearRegression() #object linear_model.LinearRegression saved inside a variable
    model.fit(x,y)  #fit method invoked on the variable, linear regression object/line returned
    y2=model.predict(x) #get the fitted values
    r2=model.score(x,y) #R-square returned 
    print('R-square value = ',r2)
    print('Value of B in y = Ax + B is ', model.coef_) #returns value of B in y = Ax + B
    print('Value of intercept = ' ,model.intercept_)  # returns array of value of intercept
    
    #PLOTTING REGRESSION LINE -
    mpp.scatter(x,y,color = 'YELLOW')
    mpp.plot(x,y2,color='B',linewidth=2)
    mpp.xlabel('Temperature')
    mpp.ylabel('Fuel_Price')
    mpp.show()
    return r2,model.coef_,model.intercept_
'''

# with sklearn
def mr(Xtrain,Ytrain,Xtest,Ytest):
    regr = linear_model.LinearRegression()
    regr.fit(Xtrain, Ytrain)
    intercept=regr.intercept_
    coef=regr.coef_
    print('Intercept: \n', intercept)
    print('Coefficients: \n', coef)
    # prediction with sklearn
    #print ('Predicted value: \n')
    print(regr.predict([[Xtest ,Ytest]]))
    predicted=regr.predict([[Xtest,Ytest]])
    return (intercept,coef,predicted)

'''
#ACCEPTING VALUES - 
dataframe1=pd.read_excel('C:/Users/User/Desktop/UST/Retail Sales prediction Project/Features2.xlsx',sep=',',header=0)
print(dataframe1)

# INITIALISING X AND Y -
x=dataframe1[['Store','Temperature']].astype(float)
y=dataframe1['CPI'].astype(float)
mr(x,y)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

companies = pd.read_csv("C:\\Users\\ROHIT\\Desktop\\companies1.csv")
X = companies.iloc[:,:-1].values
y = companies.iloc[:,4].values
print(companies.head(10))

sns.heatmap(companies.corr())
#plt.show()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])


onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
#print(X)

X = X[:,1:]
#print(X)

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X, y, test_size=0.4, random_state=0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X_train,y_train)

y_pred = regress.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error



'''
petrol = pd.read_csv("C:\\Users\\ROHIT\\Downloads\\Dataset\\Regression\\petrol_consumption.csv")
#print(petrol.head())
#print(petrol.shape)
#print(petrol.describe())

#Preparing the data
x = petrol.drop("Petrol_Consumption", axis = 1)
y = petrol["Petrol_Consumption"]

#Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

reading = pd.DataFrame({"actual": y_test, "Pred ": y_pred})
print(reading)
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(np.sqrt(mean_squared_error(y_pred,y_test)))
'''
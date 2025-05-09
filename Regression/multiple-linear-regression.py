import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


print("independent variables",x)
print("dependent variables",y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct  = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

print("After one hot encoding \n")
print(x,"\n")


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print("After Splitting data \n")
print("x_train",x_train,"\n")
print("x_test",x_test,"\n")
print("y_train",y_train,"\n")
print("y_test",y_test,"\n")


from sklearn.linear_model import LinearRegression;
regressor  = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
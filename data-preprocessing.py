import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/Users/gaurav/Machine learning/Machine learning A-Z/Part 1 - Data Preprocessing/Data Preprocessing/Python/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x,"\n")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct  = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

print("After one hot encoding \n")
print(x,"\n")

#Encoding dependent variables

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y,"\n")

#Splitting dataset into Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print("After Splitting data \n")
print("x_train",x_train,"\n")
print("x_test",x_test,"\n")
print("y_train",y_train,"\n")
print("y_test",y_test,"\n")

#Feature Scaling

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
x_train[:,3:] = standardScaler.fit_transform(x_train[:,3:])
x_test[:,3:] = standardScaler.fit_transform(x_test[:,3:])

print("After feature scaling \n")
print("x_train",x_train,"\n")
print("x_test",x_test,"\n")



## importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Initial x",x,"\n")

print("Initial y",y,"\n")

## Splitting the data set into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1)

print("After Splitting data \n")
print("x_train",x_train,"\n")
print("x_test",x_test,"\n")
print("y_train",y_train,"\n")
print("y_test",y_test,"\n")

## Feature scaling

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
x_train_1 = standardScaler.fit_transform(x_train)
x_test_1 = standardScaler.fit_transform(x_test)

print("After feature scaling \n")
print("x_train",x_train_1,"\n")
print("x_test",x_test_1,"\n")

## Classification

from sklearn.linear_model import  LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

## Prediction

print(classifier.predict(standardScaler.transform([[30,87000]])))

## Predicted vs Real Result

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred,),1),y_test.reshape(len(y_test),1)),1))

## making confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

print("Accuracy ",accuracy_score(y_test,y_pred))



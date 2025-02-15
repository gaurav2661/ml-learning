#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

y = y.reshape(len(y),1)
print(y)

from sklearn.preprocessing import StandardScaler
'''
the StandardScaler from sklearn.preprocessing is used to standardize (normalize)
the features (x) and target (y) variables of a dataset.
'''
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
print(x)
print(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)))


plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color = 'red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color = 'blue')
plt.title('truth or bluff (SVR)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
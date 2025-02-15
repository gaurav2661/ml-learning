import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL.ImageColor import colormap
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('truth or bluff (Linear Regression)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

print(lin_reg.predict([[6.5]]))

plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(x_poly),color = 'blue')
plt.title('truth or bluff (Polynomial Regression)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

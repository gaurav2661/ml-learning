#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')

'''

                                x = dataset.iloc[:, :-1].values
dataset.iloc[:, :-1]: The .iloc[] function is used to access a group of rows and columns by their integer positions.
The first part of the : indicates that you want all rows (: means all rows).
The second part :-1 means you want all columns except the last one. The -1 index refers to the last column, and :-1 excludes it.

So, dataset.iloc[:, :-1] selects all rows but only the input feature columns (i.e., all columns except the last one).

.values: This converts the selected DataFrame (which is dataset.iloc[:, :-1]) into a NumPy array. 
The .values attribute is used to get the underlying data from the DataFrame in the form of a NumPy array,
which is required for machine learning algorithms in most cases.


                                y = dataset.iloc[:, -1].values
dataset.iloc[:, -1]: This selects the last column (i.e., the target variable or dependent variable). 
The -1 index refers to the last column in the dataset, which in this case is the Salary column.

.values: Again, this converts the selected column (target variable) into a NumPy array.

So, y will be a NumPy array containing the salary values (dependent variable) corresponding to each input feature.
'''

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

'''
                                    test_size=0.2:
                                    
This argument specifies the proportion of the dataset that should be set aside as the test set.
In this case, test_size=0.2 means that 20% of the data will be used for testing,
and the remaining 80% will be used for training the model.

                                    random_state=1:
                                    
This is the random seed for shuffling the dataset before splitting it into training and test sets.
By setting random_state=1, you ensure that the results are reproducible.
The same split will occur every time you run the code with this random state.
If you omit it, the data will be shuffled randomly each time, which may lead to different splits each time you run the code.

                                 Purpose of train_test_split()
The purpose of using train_test_split() is to evaluate the model's performance.
You train your model using the training data (x_train, y_train) and then test it on the unseen test data (x_test, y_test).
This ensures that your model is not just memorizing the training data (which can lead to overfitting), but generalizing well to new, unseen data.

'''
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

print("After Splitting data \n")
print("x_train",x_train,"\n")
print("x_test",x_test,"\n")
print("y_train",y_train,"\n")
print("y_test",y_test,"\n")

# Training simple linear regression model on training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

'''
                                regressor.fit(x_train, y_train)
.fit(): This is a method that trains the model using the provided data.
 It adjusts the parameters (coefficients) of the model to best fit the training data.

x_train: These are the input features (independent variables) from the training dataset. 
The model will use this data to learn the relationship between the input features and the target variable.

y_train: These are the target values (dependent variable) corresponding to the training dataset.
 These are the values that the model is trying to predict based on the input features (x_train).
'''
regressor.fit(x_train,y_train)

# Predicting the test set result

'''
                                regressor.predict(x_test)
regressor: This is the Linear Regression model that has already been trained on the training data using the 
.fit() method (as explained earlier). After fitting the model with x_train and y_train,
 the model has learned the relationship between the input features and the target variable.

.predict(): This method is used to make predictions based on the input data.
 When you call predict() on the model, it uses the parameters (coefficients) that were learned during training to generate predicted values for the input data.

x_test: These are the input features (independent variables) from the test dataset.
 The test dataset contains new, unseen data, which the model hasn't encountered during training.
  x_test is typically a portion of the data that was split off from the original dataset to evaluate the performance of the model after training.
  
                                 What happens when you call regressor.predict(x_test)?
The trained model takes the input features from x_test (such as years of experience in a salary prediction task) and
 applies the learned coefficients (slope w and intercept b) to predict the target variable (e.g., salary).

For a simple linear regression model, the prediction for each test example can be computed using the equation:

y(pred)=w⋅x + b

where:

y_pred is the predicted salary,
x is the years of experience (input feature),
w is the learned weight (slope),
b is the learned intercept.

For multiple linear regression, where you have multiple features, the model makes predictions by applying the learned weights to each feature. The equation becomes:

𝑦(pred)=𝑤1⋅𝑥1 + 𝑤2⋅𝑥2 +⋯+ 𝑤𝑛⋅𝑥𝑛 + b
where each x_n is one of the features in the test set, and each w_n is the corresponding learned coefficient for that feature.
'''
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

# Visualize results of training set

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,x_pred,color = 'blue')
plt.title('Salary vs Experience of Training set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualize results of test set

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,x_pred,color = 'blue')
plt.title('Salary vs Experience of Test set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()




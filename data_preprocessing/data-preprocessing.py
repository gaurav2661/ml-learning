import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

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


print(x)
print(y)
'''
This imports the SimpleImputer class from sklearn.
impute, which is used to handle missing values in 
datasets by replacing them with specified values (such as the mean, median, or most frequent value).
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print("After one Imputer \n")
print(x,"\n")

'''
missing_values=np.nan: This specifies that the missing values in the dataset are represented by NaN (Not a Number).
strategy='mean': This defines how the missing values will be imputed (filled). In this case, it will replace the missing values with the mean of the non-missing values in that column.

x[:, 1:3] is a subset of the dataset x, where you're selecting all rows and columns 1 and 2 (indexing in Python is 0-based, so columns 1 and 2 correspond to the second and third columns in x).
fit() is applied to the selected columns, meaning that the imputer is calculating the mean of the existing (non-missing) values in those columns. The imputer "learns" the mean for each column to later use it for imputing the missing values.

transform() uses the learned means to replace the missing values in the selected columns (columns 1 and 2) with the computed mean.
The transformed data (with the missing values replaced) is then assigned back to x[:, 1:3], effectively replacing the original subset with the imputed data.
'''

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct  = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

'''

transformers=[('encoder', OneHotEncoder(), [0])]: This defines the transformation you want to apply.
'encoder': This is just a name given to the transformation for identification (can be anything).
OneHotEncoder(): This specifies that you want to apply one-hot encoding to the specified columns.
[0]: This indicates that you want to apply the transformation (one-hot encoding) to the first column (index 0) of the dataset x.
remainder='passthrough': This tells the ColumnTransformer to leave all other columns (columns not listed for transformation)
 unchanged and pass them through without modification. So, any columns that are not the first column will be retained as they are.



fit_transform(x): This first fits the transformer (learns the encoding scheme for the first column) and then transforms the dataset.
For the first column (index 0), OneHotEncoder will convert each unique value into a separate binary column (a one-hot encoding scheme).
The remainder='passthrough' option ensures that the other columns (those not specified for transformation) are retained unchanged.
np.array(): This converts the result back into a NumPy array. Since ColumnTransformer returns a sparse matrix by default, this step ensures the result is a regular NumPy array for easier manipulation in subsequent steps.

'''

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



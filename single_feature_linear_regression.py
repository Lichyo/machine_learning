import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv('datasets/Salary_Data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values


# fill missing values
imputer = SimpleImputer(missing_values=np.NAN, strategy='mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# no need to do one hot encoding

# split & feature scaling
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train[:, :])
# x_test = sc.transform(x_test[:, :])

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# y_pre = regressor.predict(x_test)
# plt.scatter(x_test, y_test, color='red')
# plt.plot(x_train, regressor.predict(x_train), color='blue')
# plt.title('Experience v.s. Salary')
# plt.xlabel('Years of Experience')
# plt.ylabel('Years of Salary')
# plt.show()

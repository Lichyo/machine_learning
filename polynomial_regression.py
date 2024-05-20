import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('datasets/Position_Salaries.csv')
X = data.iloc[:, 1:-1].values
Y = data.iloc[:, -1].values

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

linear_reg = LinearRegression()
linear_reg.fit(x_poly, Y)
y_predict = linear_reg.predict(x_poly)

# Polynomial Regression Plot
# plt.scatter(X, Y, color='blue')  # 散佈圖
# plt.plot(X, y_predict, color='red')  # Curve
# plt.show()

# Predict single value
# test = poly_reg.fit_transform([[12]])
# single_predict = linear_reg.predict(test)
# print(single_predict)

# Smooth one
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, Y, color='red')
# plt.plot(X_grid, linear_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

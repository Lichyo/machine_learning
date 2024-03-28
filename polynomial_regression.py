import matplotlib.pyplot as plt
import pandas as pd
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

plt.scatter(X, Y, color='blue')  # 散佈圖
plt.plot(X, y_predict, color='red')  # Curve
plt.show()

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

data = pd.read_csv('../datasets/Data.csv')
raw_x = data.iloc[:, :-1]
raw_y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, random_state=0)
raw_x_train = x_train
raw_y_train = y_train

#  multiple linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)
y_pred = linear_regressor.predict(x_test)
r2 = r2_score(y_pred, y_test)
print(f'multiple linear r2_score : {r2}')

# polynomial regression
x_train = raw_x_train
y_train = raw_y_train
y_train = np.array(y_train).reshape(len(y_train), 1)

poly_regressor = PolynomialFeatures(degree=4)
x_train = poly_regressor.fit_transform(x_train)
linear_regressor.fit(x_train, y_train)
y_pred = linear_regressor.predict(poly_regressor.transform(x_test))
r2 = r2_score(y_test, y_pred)
print(f'polynomial regression r2_score : {r2}')

# svr
svr_regressor = SVR(kernel='rbf')
x_train = raw_x_train
y_train = raw_y_train
y_train = np.array(y_train).reshape(len(y_train), 1)
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)
svr_regressor.fit(x_train, y_train)
y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(x_test)).reshape(-1, 1))
r2 = r2_score(y_test, y_pred)
print(f'SVR r2_score : {r2}')

# # decision tree
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(raw_x_train, raw_y_train)
y_pred = decision_tree_regressor.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f'Decision Tree r2_score : {r2}')

# random forest
random_forest = RandomForestRegressor(n_estimators=500)
random_forest.fit(raw_x_train, raw_y_train)
y_pred = random_forest.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f'Random Forest r2_score : {r2}')
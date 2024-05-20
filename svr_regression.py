from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('datasets/Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(x, y)
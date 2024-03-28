import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import dataset
dataset = pd.read_csv('datasets/Data.csv')
features = dataset.iloc[:, :-1].values  # index locating ( row & column )
dependent_variable = dataset.iloc[:, -1].values

# fill missing data with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(features[:, 1:3])  # fit imputer with data
features[:, 1:3] = imputer.transform(features[:, 1:3])  # replace missing data

# one hot encoding
cd = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
features = np.array(cd.fit_transform(features))

# label encoding ->
le = LabelEncoder()
dependent_variable = np.array(le.fit_transform(dependent_variable))

# split training & test set
x_train, x_test, y_train, y_test = train_test_split(features, dependent_variable, test_size=0.2)
# print(len(x_train))
# print(len(x_test))

# feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])

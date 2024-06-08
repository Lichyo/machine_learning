import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import tools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score

data = pd.read_csv('../datasets/HW2_heart.csv')
raw = data

# print(data.info()) 顯示資料有無缺失和各特徵資料型態

# find outliers
age = tools.find_and_plot_outliers(raw, 'Age')
restingBP = tools.find_and_plot_outliers(raw, 'RestingBP')
raw = raw.drop(449)  # drop the specific data

cholesterol = tools.find_and_plot_outliers(raw, 'Cholesterol')
imputer = SimpleImputer(missing_values=0, strategy='mean')
raw['Cholesterol'] = imputer.fit_transform(np.array(raw['Cholesterol']).reshape(-1, 1))

maxHR = tools.find_and_plot_outliers(raw, 'MaxHR')
Oldpeak = tools.find_and_plot_outliers(raw, 'Oldpeak')
raw = tools.delete_outliers(raw, 'Oldpeak', Oldpeak)

# tools.plot_scatter(raw, 'Age', 'HeartDisease')
# ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
# tools.heatmap(raw.drop(columns=['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']))

# 敘述性統計分析 - CSV檔
# describe = raw.describe()
# describe.to_csv('describe.csv')

# Encoding Session
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])],
    remainder='passthrough')
data = ct.fit_transform(raw)

x = data[:, :-1]
y = data[:, -1]

# split data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

new_feature_names = ct.get_feature_names_out()
numerical_features = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
columns_to_scale = [new_feature_names.tolist().index(f'remainder__{col}') for col in numerical_features]

# standard
sc = StandardScaler()
x_train[:, columns_to_scale] = sc.fit_transform(x_train[:, columns_to_scale])
x_test[:, columns_to_scale] = sc.transform(x_test[:, columns_to_scale])

# print(data.describe())
# building classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn_classifier.fit(x_train, y_train)

logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train, y_train)

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(x_train, y_train)

random_forest_classifier = RandomForestClassifier(criterion='entropy', n_estimators=10)
random_forest_classifier.fit(x_train, y_train)

linear_SVM = SVC(kernel='linear')
non_linear_SVM = SVC(kernel='rbf')
linear_SVM.fit(x_train, y_train)
non_linear_SVM.fit(x_train, y_train)

gaussianNB_classifier = GaussianNB()
gaussianNB_classifier.fit(x_train, y_train)


def score(classifier, classifier_string):
    y_pred = classifier.predict(x_test)
    print(f'{classifier_string}')
    print(f'accuracy : {round(accuracy_score(y_pred, y_test), 3)}')
    print(f'recall : {round(recall_score(y_pred, y_test), 3)}')
    print(f'precision : {round(precision_score(y_pred, y_test), 3)}')
    print(f'f1_score : {round(f1_score(y_pred, y_test), 3)}')
    print('confusion matrix : ')
    print(confusion_matrix(y_pred, y_test))
    print('\n')


print('\n')
score(knn_classifier, 'KNN_classifier')
score(logistic_classifier, 'logistic_classifier')
# score(decision_tree_classifier, 'decision_tree_classifier')
score(random_forest_classifier, 'random_forest_classifier')
# score(linear_SVM, 'linear_SVM')
score(non_linear_SVM, 'non_linear_SVM')
# score(gaussianNB_classifier, 'gaussianNB_classifier')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.metrics import classification_report

data = pd.read_csv('../datasets/HW2_heart.csv')
y = data.iloc[:, -1].values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])], remainder='passthrough')
data = np.array(ct.fit_transform(data))
x = data[:, :-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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
    print(f'{classifier_string} : ')
    print(confusion_matrix(y_pred, y_test))
    print(f'{accuracy_score(y_pred, y_test)}\n')


score(knn_classifier, 'KNN_classifier')
score(logistic_classifier, 'logistic_classifier')
score(decision_tree_classifier, 'decision_tree_classifier')
score(random_forest_classifier, 'random_forest_classifier')
score(linear_SVM, 'linear_SVM')
score(non_linear_SVM, 'non_linear_SVM')
score(gaussianNB_classifier, 'gaussianNB_classifier')

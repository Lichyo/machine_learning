import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest


def heatmap(corr):
    ax = sns.heatmap(
        corr, vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()


def pointer(x_train, x_test, y_train, y_test, regressor):
    y_train_pred = regressor.predict(x_train)
    y_test_pred = regressor.predict(x_test)
    print('MSE(training): %.3f, MSE(testing): %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred))
          )
    print('R^2(training): %.3f, R^2(testing): %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred))
          )


def scatter_with_feature_and_price(feature, price, feature_in_string):
    plt.scatter(feature, price, color='blue')
    plt.title(f'{feature_in_string} v.s. price')
    plt.xlabel(f'{feature_in_string}')
    plt.ylabel('price')
    plt.show()


def clean_data_with_isolation_forest(data):
    clf = IsolationForest(max_samples=data.shape[0], contamination=0.01)
    clf.fit(data)
    y_pred_train = clf.predict(data)
    idx = np.where(y_pred_train == -1)[0]
    outliers = data.take(idx)
    return data.drop(labels=outliers.index)

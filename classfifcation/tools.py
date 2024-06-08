import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_and_plot_outliers(df, column_name):
    data = df[column_name]
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # print(outliers.to_string())  可以得到離群值的 list
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(data, vert=False)
    # plt.scatter(outliers, np.ones(len(outliers)), color='red')
    # plt.title('Boxplot with Outliers')
    # plt.xlabel(column)
    # plt.show()
    return outliers


def delete_outliers(df, column_name, outliers):
    df_cleaned = df[~df[column_name].isin(outliers)]
    return df_cleaned


def plot_scatter(data, x, y):
    data[x].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8), cmap='Set3')
    plt.title(f'Distribution of {x}')
    plt.ylabel('')
    plt.show()
    # plt.scatter(data[x], data[y], color='red')
    # plt.title(f'{x} v.s. {y}')
    # plt.xlabel(f'{x}')
    # plt.ylabel(f'{y}')
    # plt.show()


def heatmap(data):
    corr = data.corr()
    mask = np.ones(corr.shape).astype(bool)
    mask = np.triu(mask, k=1)
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

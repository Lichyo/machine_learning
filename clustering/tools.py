import numpy as np
import matplotlib.pyplot as plt


def find_and_plot_outliers(df, column):
    data = df[column]
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.1 * IQR
    upper_bound = Q3 + 1.1 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # print(outliers.to_string())  # 可以得到離群值的 list
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import shapiro
import tools
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('HW1_house_data.csv')
df = df.drop(columns=['date', 'id'])

# Step 1 : 資料預處理
# Convert non-numeric values to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(df.isnull().sum())  # figure 1-1 沒有缺失值

# Box plots to outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()  # figure 1-2

# for col in df.columns:
#     if col == 'price':
#         continue
#     else:
#         plt.scatter(df[col], price, color='blue')
#         plt.title(f'{col} v.s. price')
#         plt.xlabel(f'{col}')
#         plt.ylabel('price')
#         plt.show()  # figure 1-3

# Step 2
data_des = df.describe()
data_des.to_csv('data_describe.csv')  # figure 2-1
price = df['price']

corr = df.corr()
# tools.heatmap(corr)  # Figure 3-1

sqft_living = df['sqft_living']
df = df.drop(
    columns=['sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_basement', 'yr_built', 'yr_renovated',
             'zipcode', 'lat', 'long', 'sqft_lot15', 'bedrooms'], axis=1)
corr = df.corr()  # person
tools.heatmap(corr)  # Figure 3-2
# corr_spearman = data.corr(method='spearman')
# corr_kendall = data.corr(method='kendall')


mask = np.ones(corr.shape).astype(bool)
mask = np.triu(mask, k=1)
upper = corr.where(mask)

# delete 特徵間相關係數太大的值
to_drop = [c for c in upper if any(upper[c] > 0.8)]
df = df.drop(df[to_drop], axis=1)

corr = df.corr()
tools.heatmap(corr)  # Figure 3-3

# scatter_with_feature_and_price(clean_data['sqft_living'], clean_data['price'], 'sqft_living')

# single linear 分析
# def single_linear_plot(df, feature):
#     y = np.array(df['price']).reshape(-1, 1)
#     x = df[feature]
#     x = np.array(x).reshape(-1, 1)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#     regressor = LinearRegression()
#     regressor.fit(x_train, y_train)
#
#     y_pre = regressor.predict(x_test)
#     plt.scatter(x_test, y_test, color='red')
#     plt.plot(x_train, regressor.predict(x_train), color='blue')
#     plt.title(f'{feature} v.s. price')
#     plt.xlabel(f'{feature}')
#     plt.ylabel('price')
#     plt.show()
#
#
# for col in df.columns:
#     if col == 'price':
#         continue
#     else:
#         single_linear_plot(df, col)  # figure 4-1

y = np.array(df['price']).reshape(-1, 1)
x = df.drop('price', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pre = regressor.predict(x_test)

r2 = r2_score(y_test, y_pre)
mse = mean_squared_error(y_test, y_pre)
print(f'R2 score: {r2}')
print(f'MSE: {mse}')  # figure 4-2

x = sm.add_constant(x)  # 增加常數行作為截距項
print(x)
model = sm.OLS(y, x).fit()

# Shapiro-Wilk 常態性檢定
stat, p = shapiro(model.resid)
print('Statistics: %.3f, p-value: %.3f' % (stat, p))
alpha = 0.05

if p > alpha:
    print('看起來是常態分布（無法拒絕H0）')
else:
    print('看起來不是常態分布（拒絕H0）')
    # figure 6-1

qqplot(model.resid, line='s')
plt.show()  # figure 6-2

dw = durbin_watson(model.resid)
print('dw: %.3f' % dw)

if 2 <= dw <= 4:
    print('誤差項獨立')
elif 0 <= dw < 2:
    print('誤差項不獨立')
else:
    print('計算錯誤')  # figure 6-3

df_resid = pd.DataFrame()
df_resid['y_pred'] = model.predict(x)
df_resid['resid'] = model.resid
df_resid = StandardScaler().fit_transform(df_resid)

kws = {'color':'red', 'lw':3}
sns.residplot(x=df_resid[:, 0], y=df_resid[:, 1],
              lowess=True, line_kws=kws)
plt.xlabel('Predicted Values (standardization)', fontsize=14)
plt.ylabel('Residual (standardization)', fontsize=14)
plt.show()  # figure 6-4
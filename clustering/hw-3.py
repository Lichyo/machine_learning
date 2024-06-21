from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import tools

df = pd.read_csv('../datasets/HW3_Credit Card Dataset.csv')
df = df.sample(n=3000, random_state=1)

# Step 1 : 資料預處理
# Convert non-numeric values to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Check for missing values
print(df.isnull().sum())  # figure 1-1
# Handle missing values by imputing with the mean
df.fillna(0, inplace=True)
# Check for missing values again
print(df.isnull().sum())  # figure 1-2

# 數值型資料檢視
df.hist(figsize=(15, 15))
plt.show()  # figure 1-3

# Box plots to outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()  # figure 1-4

# Correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()  # figure 1-5

# Step 2 : 敘述性相關性統計
des = df.describe()
des.to_csv('descriptive_statistics.csv')  # figure 2-1

# Step 3 : 特徵相關性分析
# Correlation matrix heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()  # figure 3-1

# 根據相關性矩陣，刪除ONEOFF_PURCHASES' PURCHASES_INSTALLMENTS_FREQUENCY' 'PURCHASES_TRX' 'INSTALLMENTS_PURCHASES
df.drop(['ONEOFF_PURCHASES', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'PURCHASES_TRX', 'INSTALLMENTS_PURCHASES'], axis=1,
        inplace=True)

# Step 4 : PCA
# 先比離群值移除
for col in df.columns:
    if col == 'CASH_ADVANCE' or col == 'CASH_ADVANCE_TRX':
        outliers = tools.find_and_plot_outliers(df, col)
        df = tools.delete_outliers(df, col, outliers)
    else:
        continue

# 因PCA容易受到特徵值的影響，因此需要先對資料進行標準化
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.drop('CUST_ID', axis=1))

pca = PCA()
pca.fit(scaled_df)

# Variance explained by each principal component
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()  # figure 4-1

explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()  # figure 4-3

# Reduce to 2 components for visualization
pca_2 = PCA(n_components=2)
pca_2_df = pca_2.fit_transform(scaled_df)

plt.figure(figsize=(10, 5))
plt.scatter(pca_2_df[:, 0], pca_2_df[:, 1])
plt.title('PCA - 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()  # figure 4-2

# Step 5 : K-means Clustering
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
    kmeans.fit(pca_2_df)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()  # figure 5-1

# 記錄每個K值的Silhouette Score
silhouette_scores = []

# 嘗試不同的K值，從2到10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(pca_2_df)
    score = silhouette_score(pca_2_df, y_kmeans)
    silhouette_scores.append(score)
    print(f'K={k}, Silhouette Score: {score}')

# 繪製Silhouette Score隨K值變化的圖
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()  # figure 5-2-1

# 跑單一 kmeans 3 的效果最好，達到0.5152
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(pca_2_df)

# 假設 y_kmeans 是聚類標籤的 ndarray
cluster_sizes = Counter(y_kmeans)

# 輸出每個聚類的大小
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} samples")  # Figure 5-5

# Calculate silhouette score and adjusted Rand index
silhouette_avg = silhouette_score(pca_2_df, y_kmeans)
print('Silhouette Score for K-means:', silhouette_avg)

# Plotting K-means clusters
plt.figure(figsize=(10, 5))
plt.scatter(pca_2_df[:, 0], pca_2_df[:, 1], c=y_kmeans, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()  # figure 5-3

plt.scatter(pca_2_df[y_kmeans == 0, 0], pca_2_df[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(pca_2_df[y_kmeans == 1, 0], pca_2_df[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(pca_2_df[y_kmeans == 2, 0], pca_2_df[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()  # figure 5-4

# Step 6 : Hierarchical Clustering
# Compute the linkage matrix
linked = linkage(pca_2_df, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()  # figure 6-1

# Assuming the optimal number of clusters is 4 based on the dendrogram
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(pca_2_df)

# Calculate silhouette score and adjusted Rand index
silhouette_avg = silhouette_score(pca_2_df, agglo_labels)
print('Silhouette Score for Agglomerative Clustering:', silhouette_avg)  # figure 6-2

# Plotting Agglomerative Clustering
plt.figure(figsize=(10, 5))
plt.scatter(pca_2_df[:, 0], pca_2_df[:, 1], c=agglo_labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()  # figure 6-3

plt.scatter(pca_2_df[agglo_labels == 0, 0], pca_2_df[agglo_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(pca_2_df[agglo_labels == 1, 0], pca_2_df[agglo_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(pca_2_df[agglo_labels == 2, 0], pca_2_df[agglo_labels == 2, 1], s=100, c='green', label='Cluster 3')
plt.title('Clusters of customers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()  # figure 6-4

# step 7 DBSCAN
# Determine the optimal value for eps
nearest_neighbors = NearestNeighbors(n_neighbors=5)
neighbors = nearest_neighbors.fit(pca_2_df)
distances, indices = neighbors.kneighbors(pca_2_df)

distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.title('k-NN Distance for DBSCAN')
plt.xlabel('Data Points')
plt.ylabel('4-NN Distance')
plt.show()  # figure 7-1

# Assuming the optimal value for eps is 0.85 and min_samples is 10
dbscan = DBSCAN(eps=0.85, min_samples=10)
dbscan_labels = dbscan.fit_predict(pca_2_df)

# Calculate silhouette score and adjusted Rand index
silhouette_avg = silhouette_score(pca_2_df, dbscan_labels)
print('Silhouette Score for DBSCAN:', silhouette_avg)  # figure 7-2

# Plotting DBSCAN clusters
plt.figure(figsize=(10, 5))
plt.scatter(pca_2_df[:, 0], pca_2_df[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()  # figure 7-3

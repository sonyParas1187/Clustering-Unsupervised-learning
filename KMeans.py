import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
dataset.isnull().any()
dataset.head()
dataset.shape
dataset2 = dataset.drop(['CustomerID','Genre'], axis=1)
dataset2.head()

X = dataset2.values

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.title("check optimum no of clusters with Elbow method")
plt.show()

KM5 = KMeans(n_clusters=5, random_state=42)
y_means = KM5.fit_predict(X)
KM6 = KMeans(n_clusters=6, random_state=42)
y_means6 = KM6.fit_predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y_means == 0, 0], X[y_means == 0, 1], X[y_means == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(X[y_means == 1, 0], X[y_means == 1, 1], X[y_means == 1, 2],s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter(X[y_means == 2, 0], X[y_means == 2, 1], X[y_means == 2, 2],s = 100, c = 'green', label = 'Cluster 3')
ax.scatter(X[y_means == 3, 0], X[y_means == 3, 1], X[y_means == 3, 2],s = 100, c = 'cyan', label = 'Cluster 4')
ax.scatter(X[y_means == 4, 0], X[y_means == 4, 1], X[y_means == 4, 2],s = 100, c = 'magenta', label = 'Cluster 5')
ax.scatter(KM5.cluster_centers_[:, 0], KM5.cluster_centers_[:, 1], KM5.cluster_centers_[:, 2], s = 150, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')

plt.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y_means6 == 0, 0], X[y_means6 == 0, 1], X[y_means6 == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(X[y_means6 == 1, 0], X[y_means6 == 1, 1], X[y_means6 == 1, 2],s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter(X[y_means6 == 2, 0], X[y_means6 == 2, 1], X[y_means6 == 2, 2],s = 100, c = 'green', label = 'Cluster 3')
ax.scatter(X[y_means6 == 3, 0], X[y_means6 == 3, 1], X[y_means6 == 3, 2],s = 100, c = 'cyan', label = 'Cluster 4')
ax.scatter(X[y_means6 == 4, 0], X[y_means6 == 4, 1], X[y_means6 == 4, 2],s = 100, c = 'magenta', label = 'Cluster 5')
ax.scatter(X[y_means6 == 5, 0], X[y_means6 == 5, 1], X[y_means6 == 5, 2],s = 100, c = 'black', label = 'Cluster 5')
ax.scatter(KM6.cluster_centers_[:, 0], KM6.cluster_centers_[:, 1], KM6.cluster_centers_[:, 2], s = 150, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')

plt.legend()
plt.show()

dataset.describe()

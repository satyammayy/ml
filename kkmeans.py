import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Sample data (replace with your actual data)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 8], [10, 10], [10, 12]])

# Choose the number of clusters (k)
k = 2

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Get the cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='black')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

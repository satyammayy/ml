from sklearn.cluster import KMeans
import numpy as np

# Sample data (replace with your actual data)
data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Choose the number of clusters (k)
k = 2

# Create a KMeans object
kmeans = KMeans(n_clusters=k)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_

print("Cluster labels:", labels)
print("Cluster centroids:", centroids)

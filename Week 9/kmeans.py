import numpy as np
import matplotlib.pyplot as plt
class KMeans:
    def __init__(self, k, max_iters=1000):
        self.k = k
        self.max_iters = max_iters
    def _init_centroids(self, X):
        np.random.seed(0)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]
    def _assign_clusters(self, X, centroids):
        return np.argmin(np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
    def _update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
    def fit(self, X):
        centroids = self._init_centroids(X)
        for _ in range(self.max_iters):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return centroids, labels

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(k=2)
centroids, labels = kmeans.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
plt.show( )

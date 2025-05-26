from ClusteringAlgorithm import ClusteringAlgorithm
from sklearn.cluster import KMeans, AgglomerativeClustering
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansClusterer(ClusteringAlgorithm):
    """K-Means clustering implementation."""

    def fit(self, data, n_clusters):
        """Fit K-Means clustering to the data."""
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300, tol=1e-4)
        self.clusters = self.model.fit_predict(data)
        return self.clusters

    def get_cluster_centers(self):
        """Get K-Means cluster centers."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.cluster_centers_



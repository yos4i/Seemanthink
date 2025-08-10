from sklearn.metrics import silhouette_score
from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""

    def __init__(self, distance_metric='cosine'):
        self.distance_metric = distance_metric
        self.model = None
        self.clusters = None

    @abstractmethod
    def fit(self, data, n_clusters):
        """Fit the clustering algorithm to the data."""
        pass

    @abstractmethod
    def get_cluster_centers(self):
        """Get cluster centers/centroids."""
        pass

    def calculate_silhouette_score(self, data):
        """Calculate silhouette score for the clustering."""
        if len(set(self.clusters)) > 1:
            metric = 'cosine' if self.distance_metric == 'cosine' else 'euclidean'
            return silhouette_score(data, self.clusters, metric=metric)
        return -1

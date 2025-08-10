from datetime import datetime
try:
    from .VocabularyLoader import VocabularyLoader
    from .DataExporter import DataExporter
    from .Visualizer import Visualizer
    from .KMeansClusterer import KMeansClusterer
    from .ClusterAnalyzer import ClusterAnalyzer
    from .SemanticExplorer import SemanticExplorer
except ImportError:
    from VocabularyLoader import VocabularyLoader
    from DataExporter import DataExporter
    from Visualizer import Visualizer
    from KMeansClusterer import KMeansClusterer
    from ClusterAnalyzer import ClusterAnalyzer
    from SemanticExplorer import SemanticExplorer
import os

class VocabularyClusteringSystem:
    """Main system that orchestrates all components."""

    def __init__(self, word2vec_path, vocabulary_path, results_folder,
                 algorithm='kmeans', distance_metric='cosine', normalizer=None):
        self.loader = VocabularyLoader(word2vec_path, vocabulary_path, normalizer)
        self.exporter = DataExporter(results_folder)
        self.visualizer = Visualizer(results_folder)
        self.algorithm_name = algorithm.lower()
        self.distance_metric = distance_metric
        self.algorithm = None
        self.analyzer = None

        # Initialize clustering algorithm
        if self.algorithm_name == 'kmeans':
            self.algorithm = KMeansClusterer(distance_metric)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def load_data(self):
        """Load vocabulary and word vectors."""
        self.loader.load_vocabulary()

    def cluster(self, n_clusters):
        """Perform clustering."""
        vocabulary = self.loader.get_vocabulary()
        word_vectors = self.loader.get_word_vectors()

        print(f"Testing {self.algorithm_name.upper()} with n_clusters={n_clusters}")
        clusters = self.algorithm.fit(word_vectors, n_clusters)

        # Calculate silhouette score
        silhouette_avg = self.algorithm.calculate_silhouette_score(word_vectors)
        print(f"Silhouette Score ({self.distance_metric}): {silhouette_avg}")

        # Initialize analyzer
        self.analyzer = ClusterAnalyzer(vocabulary, word_vectors, clusters, self.algorithm)

        return clusters, silhouette_avg

    def export_results(self, n_clusters, silhouette_score, dataset_name, clusters):
        """Export all clustering results."""
        vocabulary = self.loader.get_vocabulary()

        # Save clusters
        self.exporter.save_clusters_to_csv(vocabulary, clusters, f"clusters_{n_clusters}.csv")

        # Save cluster counts
        self.exporter.save_cluster_counts_to_csv(clusters, f"cluster_counts_{n_clusters}.csv")

        # Save silhouette scores
        timestamp = datetime.now().strftime("%d-%m-%Y-%H_%M")
        scaler = self.loader.get_scaler()
        self.exporter.save_silhouette_scores(
            silhouette_score, n_clusters, dataset_name,
            self.algorithm_name, self.distance_metric,
            scaler, f'silhouette_scores_{timestamp}.csv'
        )

        # Save centroids if available
        if hasattr(self.algorithm, 'get_cluster_centers'):
            self.exporter.save_centroids_to_csv(
                self.algorithm.get_cluster_centers(), f'centroids_{n_clusters}.csv'
            )

    def create_visualizations(self, n_clusters, clusters):
        """Create all visualizations."""
        vocabulary = self.loader.get_vocabulary()
        word_vectors = self.loader.get_word_vectors()

        # Basic cluster plot
        self.visualizer.plot_clusters(vocabulary, word_vectors, clusters,
                                      self.algorithm, f"clusters_{n_clusters}.html")

        # Centroids and paths plot
        self.visualizer.plot_centroids_and_paths(vocabulary, word_vectors, clusters,
                                                 self.algorithm, "centroids_paths.html")

        # Medoids plot with distances
        if self.analyzer:
            self.visualizer.plot_medoids_and_paths_with_distances(
                vocabulary, word_vectors, clusters, self.analyzer,
                "medoids_with_distances.html", self.exporter,
                "medoid_distances.csv", self.distance_metric
            )

    def run_complete_analysis(self, n_clusters, dataset_name):
        """Run complete clustering analysis."""
        # Load data
        self.load_data()

        # Perform clustering
        clusters, silhouette_score = self.cluster(n_clusters)

        # Export results
        self.export_results(n_clusters, silhouette_score, dataset_name, clusters)

        # Create visualizations
        self.create_visualizations(n_clusters, clusters)

        return clusters, silhouette_score

    def find_word_info(self, word):
        """Get information about a specific word."""
        if self.analyzer is None:
            print("No clustering has been performed yet.")
            return None

        cluster = self.analyzer.find_word_cluster(word)
        if cluster is not None:
            distance = self.analyzer.find_distance_to_centroid(word, self.distance_metric)
            print(
                f"The word '{word}' belongs to cluster {cluster} and is {distance:.4f} distance units from its centroid.")
            return cluster, distance
        return None




# Example usage and main execution
if __name__ == "__main__":
    # Configuration
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"
    results_folder = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\cluster_results"

    # User input (you can modify this section to use input() as in original code)
    algorithm = 'kmeans'
    distance_metric = 'cosine'
    normalizer = 'standard'
    n_clusters = 10

    # Create and run the clustering system
    system = VocabularyClusteringSystem(
        word2vec_path=word2vec_path,
        vocabulary_path=vocabulary_path,
        results_folder=results_folder,
        algorithm=algorithm,
        distance_metric=distance_metric,
        normalizer=normalizer
    )

    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(vocabulary_path))[0]

    # Run complete analysis
    clusters, silhouette_score = system.run_complete_analysis(n_clusters, dataset_name)

    print(f"Analysis complete. Results saved to {results_folder}")
    #printerh
    # Start semantic exploration loop
    explorer = SemanticExplorer(
        analyzer=system.analyzer,
        word_vectors=system.loader.get_word_vectors(),
        vocabulary=system.loader.get_vocabulary(),
        full_vectors=system.loader.get_word_vectors(),
        full_vocab=system.loader.get_vocabulary(),
        full_clusters=clusters
    )

    explorer.run_interactive_exploration()
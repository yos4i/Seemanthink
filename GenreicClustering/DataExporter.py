from collections import Counter
import pandas as pd
import os


class DataExporter:
    """Handles exporting clustering results to various formats."""

    def __init__(self, results_folder):
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def save_clusters_to_csv(self, vocabulary, clusters, output_filename):
        try:
            df = pd.DataFrame({
                'word': vocabulary,
                'cluster': clusters
            })
            output_path = os.path.join(self.results_folder, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Cluster results saved to {output_path}.")
        except Exception as e:
            print(f"An error occurred while saving clusters to CSV: {e}")

    def save_cluster_counts_to_csv(self, clusters, output_filename):
        try:
            cluster_counts = Counter(clusters)
            df = pd.DataFrame({
                'cluster': list(cluster_counts.keys()),
                'count': list(cluster_counts.values())
            })
            output_path = os.path.join(self.results_folder, output_filename)
            if not os.path.isfile(output_path):
                df.to_csv(output_path, index=False)
            else:
                with open(output_path, mode='a', encoding='utf-8', newline='') as f:
                    df.to_csv(f, header=False, index=False)
            print(f"Cluster counts saved to {output_path}.")
        except Exception as e:
            print(f"An error occurred while saving cluster counts to CSV: {e}")

    def save_silhouette_scores(self, silhouette_score, n_clusters, dataset_name,
                               algorithm_name, distance_metric, scaler, output_filename):
        try:
            normalization_name = type(scaler).__name__ if scaler else 'None'
            df = pd.DataFrame({
                'Dataset': [dataset_name],
                'Normalization': [normalization_name],
                'Clustering_algorithm': [algorithm_name],
                'Distance_metric': [distance_metric],
                'n_clusters': [n_clusters],
                'silhouette_score': [silhouette_score],
            })

            output_path = os.path.join(self.results_folder, output_filename)
            if not os.path.isfile(output_path):
                df.to_csv(output_path, index=False)
            else:
                with open(output_path, mode='a', encoding='utf-8', newline='') as f:
                    df.to_csv(f, header=False, index=False)
            print(f"Silhouette score for n_clusters={n_clusters} saved to {output_path}.")
        except Exception as e:
            print(f"An error occurred while saving silhouette scores: {e}")

    def save_centroids_to_csv(self, centroids, output_filename):
        try:
            df = pd.DataFrame(centroids)
            output_path = os.path.join(self.results_folder, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Centroids saved to {output_path}.")
        except Exception as e:
            print(f"An error occurred while saving centroids: {e}")

    def save_medoid_distances_to_csv(self, medoid_distances_data, output_filename):
        try:
            df = pd.DataFrame(medoid_distances_data)
            output_path = os.path.join(self.results_folder, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Saved medoid distances CSV to {output_path}")
        except Exception as e:
            print(f"An error occurred while saving medoid distances: {e}")

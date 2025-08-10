import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import plotly.express as px

class Visualizer:
    """Handles all visualization tasks."""

    def __init__(self, results_folder):
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def plot_clusters(self, vocabulary, word_vectors, clusters, algorithm, output_filename):
        """Create scatter plot of clusters using PCA."""
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(word_vectors)

            df = pd.DataFrame({
                'word': vocabulary,
                'cluster': clusters,
                'pca1': pca_result[:, 0],
                'pca2': pca_result[:, 1]
            })

            color_scale = px.colors.qualitative.Plotly
            fig = px.scatter(df, x='pca1', y='pca2', color='cluster',
                             title=f'Clustering of Vocabulary Words (Clusters={len(set(clusters))})',
                             color_continuous_scale=color_scale,
                             labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2', 'cluster': 'Cluster'},
                             opacity=0.8)

            fig.update_traces(
                marker=dict(size=5, opacity=0.7, line=dict(width=0.1, color='DarkSlateGrey')),
                selector=dict(mode='markers')
            )

            # Add centroids if available
            if hasattr(algorithm, 'get_cluster_centers'):
                centroids = pca.transform(algorithm.get_cluster_centers())
                fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1],
                                mode='markers',
                                marker=dict(color='black', size=8, symbol='circle'),
                                name='Cluster Centers')

            fig.update_layout(
                legend_title='Cluster',
                title_font_size=20,
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='closest'
            )

            output_path = os.path.join(self.results_folder, output_filename)
            fig.write_html(output_path)
            print(f"Cluster plot successfully saved to {output_path}.")

        except Exception as e:
            print(f"An error occurred while plotting clusters: {e}")

    def plot_silhouette_scores(self, silhouette_csv_path, output_filename):
        """Plot silhouette scores vs number of clusters."""
        try:
            df = pd.read_csv(silhouette_csv_path)
            df = df[pd.to_numeric(df['n_clusters'], errors='coerce').notnull()]
            df['n_clusters'] = df['n_clusters'].astype(int)
            df = df[df['silhouette_score'] >= -1]

            fig = px.line(df, x='n_clusters', y='silhouette_score',
                          title='Silhouette Score vs Number of Clusters')

            output_path = os.path.join(self.results_folder, output_filename)
            fig.write_html(output_path)
            print(f"Silhouette score plot saved to {output_path}.")

        except Exception as e:
            print(f"An error occurred while plotting silhouette scores: {e}")

    def plot_centroids_and_paths(self, vocabulary, word_vectors, clusters, algorithm, output_filename):
        """Plot clusters with centroids and paths between centroids."""
        try:
            if not hasattr(algorithm, 'get_cluster_centers'):
                print("Centroids are not available for this algorithm.")
                return

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(word_vectors)
            centroids_2d = pca.transform(algorithm.get_cluster_centers())

            df = pd.DataFrame({
                'word': vocabulary,
                'cluster': clusters,
                'pca1': pca_result[:, 0],
                'pca2': pca_result[:, 1]
            })

            color_scale = px.colors.qualitative.Plotly
            fig = px.scatter(df, x='pca1', y='pca2', color='cluster',
                             title=f'Clustering with Centroids (Clusters={len(set(clusters))})',
                             color_continuous_scale=color_scale,
                             opacity=0.35)

            # Add centroids
            fig.add_scatter(
                x=centroids_2d[:, 0],
                y=centroids_2d[:, 1],
                mode='markers+text',
                marker=dict(color='black', size=25, symbol='circle',
                            line=dict(width=5, color='blue')),
                text=[f'C{i}' for i in range(len(centroids_2d))],
                textposition='top center',
                name='Centroids'
            )

            # Add paths between centroids
            num_clusters = len(centroids_2d)
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    fig.add_scatter(
                        x=[centroids_2d[i, 0], centroids_2d[j, 0]],
                        y=[centroids_2d[i, 1], centroids_2d[j, 1]],
                        mode='lines',
                        line=dict(color='black', width=2.5, dash='solid'),
                        showlegend=False
                    )

            fig.update_layout(
                title_font_size=20,
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='closest'
            )

            output_path = os.path.join(self.results_folder, output_filename)
            fig.write_html(output_path)
            print(f"Centroids and paths plot saved to {output_path}.")

        except Exception as e:
            print(f"An error occurred while plotting centroids and paths: {e}")

    def plot_medoids_and_paths_with_distances(self, vocabulary, word_vectors, clusters,
                                              analyzer, output_html_filename, exporter,
                                              output_csv_filename, distance_metric='cosine'):
        """Plot medoids with distance information."""
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(word_vectors)

            df = pd.DataFrame({
                'word': vocabulary,
                'cluster': clusters,
                'pca1': pca_result[:, 0],
                'pca2': pca_result[:, 1]
            })

            # Get medoids
            medoids = analyzer.get_cluster_medoids(distance_metric)
            medoid_ids = sorted(medoids.keys())
            medoid_words = [medoids[cid] for cid in medoid_ids]
            medoid_vectors = np.array([word_vectors[vocabulary.index(word)] for word in medoid_words])
            medoids_2d = pca.transform(medoid_vectors)

            # Create plot
            color_scale = px.colors.qualitative.Plotly
            fig = px.scatter(df, x='pca1', y='pca2', color='cluster',
                             title=f'Clustering with Medoids (Clusters={len(set(clusters))})',
                             color_continuous_scale=color_scale,
                             opacity=0.35)

            # Add medoids
            fig.add_scatter(
                x=medoids_2d[:, 0],
                y=medoids_2d[:, 1],
                mode='markers+text',
                marker=dict(color='black', size=25, symbol='circle',
                            line=dict(width=5, color='blue')),
                text=[f'M{cid}' for cid in medoid_ids],
                textposition='top center',
                name='Medoids'
            )

            # Calculate distances and add paths
            records = []
            for i in range(len(medoid_ids)):
                for j in range(i + 1, len(medoid_ids)):
                    id1, id2 = medoid_ids[i], medoid_ids[j]
                    word1, word2 = medoid_words[i], medoid_words[j]

                    vec1 = medoid_vectors[i].reshape(1, -1)
                    vec2 = medoid_vectors[j].reshape(1, -1)

                    eu_dist = euclidean_distances(vec1, vec2)[0][0]
                    cos_dist = cosine_distances(vec1, vec2)[0][0]

                    hover_text = f"Medoid {id1} ({word1}) ↔ {id2} ({word2})<br>Euclidean: {eu_dist:.4f}<br>Cosine: {cos_dist:.4f}"

                    fig.add_scatter(
                        x=[medoids_2d[i, 0], medoids_2d[j, 0]],
                        y=[medoids_2d[i, 1], medoids_2d[j, 1]],
                        mode='lines',
                        line=dict(color='black', width=2.5),
                        hoverinfo='text',
                        hovertext=hover_text,
                        showlegend=False
                    )

                    records.append({
                        'Medoid1_ID': id1,
                        'Medoid1_Word': word1,
                        'Medoid2_ID': id2,
                        'Medoid2_Word': word2,
                        'Euclidean_Distance': eu_dist,
                        'Cosine_Distance': cos_dist
                    })

            # Save distances to CSV
            exporter.save_medoid_distances_to_csv(records, output_csv_filename)

            # Save plot
            fig.update_layout(
                title_font_size=20,
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='closest'
            )

            output_path = os.path.join(self.results_folder, output_html_filename)
            fig.write_html(output_path)
            print(f"Medoids and paths plot saved to {output_path}.")

        except Exception as e:
            print(f"An error occurred: {e}")

# semantle_semantic_explorer.py
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import numpy as np
import os
import pandas as pd
from datetime import datetime
try:
    from .KMeansClusterer import KMeansClusterer
    from .Visualizer import Visualizer
    from .ClusterAnalyzer import ClusterAnalyzer
except ImportError:
    from KMeansClusterer import KMeansClusterer
    from Visualizer import Visualizer
    from ClusterAnalyzer import ClusterAnalyzer

class SemanticExplorer:
    def __init__(self, analyzer, word_vectors, vocabulary,
                 full_vectors=None, full_vocab=None, full_clusters=None,
                 base_output_folder="cluster_results"):
        self.analyzer = analyzer
        self.word_vectors = word_vectors
        self.vocabulary = vocabulary
        self.guess_history = []

        self.full_vectors = full_vectors if full_vectors is not None else word_vectors
        self.full_vocab = full_vocab if full_vocab is not None else vocabulary
        self.full_clusters = full_clusters

        if base_output_folder.endswith("run_initial"):
            self.output_folder = base_output_folder
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = os.path.join(base_output_folder, f"run_{timestamp}")
            os.makedirs(self.output_folder, exist_ok=True)

    def get_medoid_recommendations(self):
        return self.analyzer.get_cluster_medoids()

    def get_vector(self, word):
        return self.word_vectors[self.vocabulary.index(word)]

    def request_score_from_user(self, word):
        print(f"Guess this word in Semantle: '{word}'")
        while True:
            try:
                score = float(input("Enter the score : "))
                if -100 <= score <= 100:
                    return score
            except ValueError:
                print("Invalid input. Please enter a number.")

    def distance(self, word1, word2):
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        return euclidean_distances([vec1], [vec2])[0][0]

    def suggest_next_medoid(self, medoids):
        if not self.guess_history:
            return list(medoids.values())[0]

        if len(self.guess_history) >= len(medoids):
            top_guess = max(self.guess_history, key=lambda x: x['score'])
            print("✅ All medoids visited. Returning highest scoring so far.")
            return top_guess['word']

        last_guess = self.guess_history[-1]
        last_word = last_guess['word']
        last_score = last_guess['score']
        last_vec = self.get_vector(last_word)

        remaining = [w for w in medoids.values() if w not in [g['word'] for g in self.guess_history]]

        ranked = []
        for word in remaining:
            vec = self.get_vector(word)
            dist = euclidean_distances([last_vec], [vec])[0][0]
            ranked.append((word, dist))

        ranked.sort(key=lambda x: x[1], reverse=(last_score <= 0))
        return ranked[0][0] if ranked else None

    def refine_cluster_data(self, target_cluster_id, full_vocab, full_vectors, full_clusters):
        cluster_words = [word for word, cluster in zip(full_vocab, full_clusters) if cluster == target_cluster_id]
        cluster_vectors = np.array([full_vectors[full_vocab.index(word)] for word in cluster_words])
        return cluster_words, cluster_vectors

    def recursive_clustering(self, sub_vocab, sub_vectors, depth=1):
        print(f"\n🔬 Recursive Clustering Level {depth}...")
        if len(sub_vocab) < 3:
            print("⛔ Not enough words to re-cluster.")
            return

        n_clusters = min(10, len(sub_vocab))
        kmeans = KMeansClusterer(distance_metric='euclidean')
        clusters = kmeans.fit(sub_vectors, n_clusters)
        sil_score = kmeans.calculate_silhouette_score(sub_vectors)

        print(f"Sub-clustering complete (k={n_clusters}) | Silhouette Score: {sil_score:.4f}\n")

        new_analyzer = ClusterAnalyzer(sub_vocab, sub_vectors, clusters, kmeans)

        # Save CSV of clusters
        cluster_df = pd.DataFrame({"word": sub_vocab, "cluster": clusters})
        csv_path = os.path.join(self.output_folder, f"sub_clusters_level{depth}.csv")
        cluster_df.to_csv(csv_path, index=False)

        # Save visualizations
        visualizer = Visualizer(self.output_folder)
        visualizer.plot_clusters(sub_vocab, sub_vectors, clusters, kmeans, f"clusters_level{depth}.html")
        visualizer.plot_centroids_and_paths(sub_vocab, sub_vectors, clusters, kmeans, f"centroids_paths_level{depth}.html")
        visualizer.plot_medoids_and_paths_with_distances(
            sub_vocab, sub_vectors, clusters, new_analyzer,
            f"medoids_level{depth}.html", exporter=None, output_csv_filename="", distance_metric="euclidean")

        sub_explorer = SemanticExplorer(
            new_analyzer, sub_vectors, sub_vocab,
            full_vectors=sub_vectors, full_vocab=sub_vocab, full_clusters=clusters,
            base_output_folder=self.output_folder
        )
        sub_explorer.run_interactive_exploration()

    def run_interactive_exploration(self):
        medoids = self.get_medoid_recommendations()

        while True:
            word = self.suggest_next_medoid(medoids)
            if word is None:
                print("✅ No more suggestions available.")
                break

            score = self.request_score_from_user(word)
            cluster = self.analyzer.find_word_cluster(word)

            self.guess_history.append({
                'word': word,
                'score': score,
                'cluster': cluster
            })

            print(f"✔️ Recorded: '{word}' | Cluster {cluster} | Score {score}\n")

            if score >= 23:
                print(f"High score! Refining cluster {cluster} now...\n")
                sub_vocab, sub_vectors = self.refine_cluster_data(
                    cluster, self.full_vocab, self.full_vectors, self.full_clusters)
                print(f"🔁 Cluster {cluster} contains {len(sub_vocab)} words. Starting recursive exploration...")
                self.recursive_clustering(sub_vocab, sub_vectors, depth=1)
                break

import numpy as np

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class ClusterAnalyzer:
    """Analyzes clustering results and provides insights."""

    def __init__(self, vocabulary, word_vectors, clusters, algorithm):
        self.vocabulary = vocabulary
        self.word_vectors = word_vectors
        self.clusters = clusters
        self.algorithm = algorithm

    def find_word_cluster(self, word):
        """Find which cluster a word belongs to."""
        if word not in self.vocabulary:
            print(f"Word {word} not found in vocabulary.")
            return None
        index = self.vocabulary.index(word)
        return self.clusters[index]

    def find_distance_to_centroid(self, word, distance_metric='cosine'):
        """Find distance from word to its cluster centroid."""
        if not hasattr(self.algorithm, 'get_cluster_centers'):
            print("Centroids are not available for this algorithm.")
            return None

        if word not in self.vocabulary:
            print(f"Word {word} not found in vocabulary.")
            return None

        index = self.vocabulary.index(word)
        vector = self.word_vectors[index]
        cluster = self.clusters[index]
        centroid = self.algorithm.get_cluster_centers()[cluster]

        if distance_metric == 'cosine':
            dist = cosine_distances([vector], [centroid])[0][0]
        else:
            dist = euclidean_distances([vector], [centroid])[0][0]
        return dist

    def get_centroid_words(self, cluster_id, n_words=5):
        """Get words closest to cluster centroid."""
        if not hasattr(self.algorithm, 'get_cluster_centers'):
            raise ValueError("Centroids are not available for this algorithm.")

        centroid = self.algorithm.get_cluster_centers()[cluster_id]
        words_in_cluster = [w for w in self.vocabulary if self.find_word_cluster(w) == cluster_id]

        # Sort by distance to centroid
        words_sorted = sorted(words_in_cluster,
                              key=lambda w: self.find_distance_to_centroid(w))
        return words_sorted[:n_words]

    def get_cluster_medoids(self, distance_metric='cosine'):
        """Get medoid (closest word to centroid) for each cluster."""
        if not hasattr(self.algorithm, 'get_cluster_centers'):
            raise ValueError("Medoids are only available for algorithms with centroids.")

        medoids = {}
        for cluster_id in np.unique(self.clusters):
            centroid = self.algorithm.get_cluster_centers()[cluster_id]
            words_in_cluster = [word for word, cluster in zip(self.vocabulary, self.clusters)
                                if cluster == cluster_id]

            vectors_in_cluster = np.array([self.word_vectors[self.vocabulary.index(word)]
                                           for word in words_in_cluster])

            if distance_metric == 'cosine':
                distances = cosine_distances([centroid], vectors_in_cluster)[0]
            else:
                distances = euclidean_distances([centroid], vectors_in_cluster)[0]

            min_index = np.argmin(distances)
            medoid_word = words_in_cluster[min_index]
            medoids[cluster_id] = medoid_word

        return medoids


    def suggest_next_medoid(medoids, word_vectors, vocabulary, previous_guesses):
        # קבל את המדואיד עם הניקוד הכי גבוה עד עכשיו
        top_guess = max(previous_guesses, key=lambda x: x['score'])
        top_cluster = top_guess['cluster']

        # סנן החוצה את המדואידים שנבדקו
        remaining = [m for cid, m in medoids.items() if m not in [g['word'] for g in previous_guesses]]

        # מצא מדואיד רחוק מהמדואיד הנוכחי (אוקלידית או קוסינוסית)
        current_vector = word_vectors[vocabulary.index(top_guess['word'])]
        farthest = None
        max_dist = -1

        for word in remaining:
            vec = word_vectors[vocabulary.index(word)]
            dist = cosine_distances([current_vector], [vec])[0][0]
            if dist > max_dist:
                max_dist = dist
                farthest = word

        return farthest

import numpy as np
from scipy import spatial
from gensim.models import KeyedVectors
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
from collections import defaultdict

# Import your existing classes
from VocabularyClusteringSystem import VocabularyClusteringSystem
from SemanticExplorer import SemanticExplorer
from ClusterAnalyzer import ClusterAnalyzer
from KMeansClusterer import KMeansClusterer


class AutomatedSemantleSolver:
    """
    Automated Semantle solver using clustering strategy.
    Combines the simulator with clustering system for intelligent guessing.
    """

    def __init__(self, word2vec_path: str, vocabulary_path: str, results_folder: str = "semantle_automated_results"):
        """
        Initialize the automated solver.

        Args:
            word2vec_path: Path to Word2Vec model
            vocabulary_path: Path to vocabulary file
            results_folder: Folder to save results
        """
        # Initialize clustering system
        self.clustering_system = VocabularyClusteringSystem(
            word2vec_path=word2vec_path,
            vocabulary_path=vocabulary_path,
            results_folder=results_folder,
            algorithm='kmeans',
            distance_metric='cosine',
            normalizer='standard'
        )

        # Load Word2Vec for similarity calculations
        print("Loading Word2Vec model for similarity calculations...")
        self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        # Initialize tracking variables
        self.target_word = None
        self.guess_history = []
        self.cluster_exploration = defaultdict(list)  # Track which clusters we've explored

        # Results storage
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate Semantle-style similarity between two words."""
        if word1 not in self.wv or word2 not in self.wv:
            return -1.0

        similarity = (1 - spatial.distance.cosine(
            self.wv[word1],
            self.wv[word2]
        )) * 100

        return round(similarity, 2)

    def solve_with_clustering(self, target_word: str, n_clusters: int = 10,
                              visualize: bool = True, verbose: bool = True) -> Dict:
        """
        Solve Semantle using clustering strategy.

        Args:
            target_word: The target word to find
            n_clusters: Number of initial clusters
            visualize: Whether to create visualizations
            verbose: Whether to print progress

        Returns:
            Dictionary with solution details
        """
        self.target_word = target_word
        self.guess_history = []
        self.cluster_exploration = defaultdict(list)

        if verbose:
            print(f"\n🎯 Target word: {target_word}")
            print(f"📊 Creating {n_clusters} initial clusters...")

        # Step 1: Load data and create initial clusters
        self.clustering_system.load_data()

        # Start recursive clustering process
        vocabulary = self.clustering_system.loader.get_vocabulary()
        word_vectors = self.clustering_system.loader.get_word_vectors()

        # Check if target word is in vocabulary
        if target_word not in vocabulary:
            print(f"❌ Target word '{target_word}' not in vocabulary!")
            return None

        # Recursive clustering function
        def recursive_cluster_search(words_to_cluster, vectors_to_cluster, depth=0, parent_cluster=""):
            """Recursively search through clusters."""

            # Limit recursion depth to avoid infinite loops
            if depth > 5 or len(words_to_cluster) < 10:
                return False

            # Create clusters
            n_sub_clusters = min(10, max(2, len(words_to_cluster) // 10))

            if verbose:
                print(
                    f"\n{'  ' * depth}📊 Level {depth}: Creating {n_sub_clusters} clusters from {len(words_to_cluster)} words...")

            sub_clusterer = KMeansClusterer('cosine')
            sub_clusters = sub_clusterer.fit(vectors_to_cluster, n_sub_clusters)

            # Create analyzer
            sub_analyzer = ClusterAnalyzer(words_to_cluster, vectors_to_cluster, sub_clusters, sub_clusterer)
            medoids = sub_analyzer.get_cluster_medoids()

            # ALWAYS test ALL medoids first
            medoid_scores = []
            if verbose:
                print(f"{'  ' * depth}🔍 Testing all {len(medoids)} medoids...")

            for cluster_id, medoid_word in medoids.items():
                score = self.calculate_similarity(medoid_word, target_word)

                cluster_name = f"{parent_cluster}{cluster_id}" if parent_cluster else str(cluster_id)
                guess_info = {
                    'word': medoid_word,
                    'score': score,
                    'cluster': cluster_name,
                    'guess_number': len(self.guess_history) + 1,
                    'type': f'medoid-level-{depth}'
                }

                self.guess_history.append(guess_info)
                medoid_scores.append((cluster_id, medoid_word, score))

                if verbose:
                    print(
                        f"{'  ' * depth}Guess {guess_info['guess_number']}: {medoid_word} (Cluster {cluster_name}) -> {score:.2f}")

                if score == 100.0:
                    if verbose:
                        print(f"\n✅ Found target word in {guess_info['guess_number']} guesses!")
                    return True

            # Sort by score to find best medoid
            medoid_scores.sort(key=lambda x: x[2], reverse=True)
            best_cluster_id, best_medoid, best_score = medoid_scores[0]

            if verbose:
                print(f"\n{'  ' * depth}📈 Best medoid: {best_medoid} with score {best_score:.2f}")

            # Now decide what to do based on best score
            if best_score > 60:
                # Very high score! Search within this cluster instead of going deeper
                if verbose:
                    print(
                        f"\n{'  ' * depth}🎯 Excellent score ({best_score:.2f})! Searching within Cluster {best_cluster_id}...")

                # Get words in best cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == best_cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]

                if verbose:
                    print(f"{'  ' * depth}📍 Searching through {len(cluster_words)} words in this cluster...")

                # Sort by distance to medoid
                words_sorted = sorted(
                    cluster_words,
                    key=lambda w: sub_analyzer.find_distance_to_centroid(w, 'cosine')
                )

                # Try many words from this cluster since we're close
                for word in words_sorted[:min(100, len(words_sorted))]:
                    if word not in [g['word'] for g in self.guess_history]:
                        score = self.calculate_similarity(word, target_word)

                        guess_info = {
                            'word': word,
                            'score': score,
                            'cluster': f"{parent_cluster}{best_cluster_id}" if parent_cluster else str(best_cluster_id),
                            'guess_number': len(self.guess_history) + 1,
                            'type': f'high-score-cluster-search-{depth}'
                        }

                        self.guess_history.append(guess_info)

                        if verbose and score > 50:  # Show promising guesses
                            print(f"{'  ' * depth}Guess {guess_info['guess_number']}: {word} -> {score:.2f}")

                        if score == 100.0:
                            if verbose:
                                print(f"\n✅ Found target word in {guess_info['guess_number']} guesses!")
                            return True

                return False  # Exhausted this cluster

            elif best_score > 23:
                # Good score but not excellent - continue with recursive clustering
                if verbose:
                    print(f"{'  ' * depth}🎯 Score > 23! Diving into Cluster {best_cluster_id}...")

                # Get words in best cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == best_cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]
                cluster_vectors = vectors_to_cluster[cluster_indices]

                # Only go deeper if we have enough words
                if len(cluster_words) >= 30:
                    # Recursive call
                    found = recursive_cluster_search(
                        cluster_words,
                        cluster_vectors,
                        depth + 1,
                        f"{parent_cluster}{best_cluster_id}-" if parent_cluster else f"{best_cluster_id}-"
                    )

                    if found:
                        return True
                else:
                    # Not enough words to cluster further - search within this cluster
                    if verbose:
                        print(f"{'  ' * depth}📍 Cluster too small ({len(cluster_words)} words). Searching within it...")

                    words_sorted = sorted(
                        cluster_words,
                        key=lambda w: sub_analyzer.find_distance_to_centroid(w, 'cosine')
                    )

                    for word in words_sorted[:min(50, len(words_sorted))]:
                        if word not in [g['word'] for g in self.guess_history]:
                            score = self.calculate_similarity(word, target_word)

                            guess_info = {
                                'word': word,
                                'score': score,
                                'cluster': f"{parent_cluster}{best_cluster_id}" if parent_cluster else str(
                                    best_cluster_id),
                                'guess_number': len(self.guess_history) + 1,
                                'type': f'small-cluster-search-{depth}'
                            }

                            self.guess_history.append(guess_info)

                            if verbose and score > 40:
                                print(f"{'  ' * depth}Guess {guess_info['guess_number']}: {word} -> {score:.2f}")

                            if score == 100.0:
                                if verbose:
                                    print(f"\n✅ Found target word in {guess_info['guess_number']} guesses!")
                                return True

            # If no medoid > 23, try exploring clusters in order of best scores
            if verbose:
                print(f"\n{'  ' * depth}📍 No medoid > 23. Exploring clusters by best score...")

            # Try top 3 clusters
            for cluster_id, medoid_word, medoid_score in medoid_scores[:3]:
                if medoid_score < 5:  # Skip very low scoring clusters
                    continue

                if verbose:
                    print(f"\n{'  ' * depth}🔎 Exploring Cluster {cluster_id} (medoid score: {medoid_score:.2f})")

                # Get words in this cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]
                cluster_vectors = vectors_to_cluster[cluster_indices]

                # If cluster is large enough, do recursive clustering
                if len(cluster_words) >= 50 and depth < 3:
                    found = recursive_cluster_search(
                        cluster_words,
                        cluster_vectors,
                        depth + 1,
                        f"{parent_cluster}{cluster_id}-" if parent_cluster else f"{cluster_id}-"
                    )

                    if found:
                        return True
                else:
                    # Try some words from this cluster
                    words_sorted = sorted(
                        zip(cluster_words, cluster_indices),
                        key=lambda x: sub_analyzer.find_distance_to_centroid(x[0], 'cosine')
                    )

                    words_tried = 0
                    for word, _ in words_sorted[:min(30, len(words_sorted))]:
                        if word not in [g['word'] for g in self.guess_history]:
                            score = self.calculate_similarity(word, target_word)

                            guess_info = {
                                'word': word,
                                'score': score,
                                'cluster': f"{parent_cluster}{cluster_id}" if parent_cluster else str(cluster_id),
                                'guess_number': len(self.guess_history) + 1,
                                'type': f'cluster-member-level-{depth}'
                            }

                            self.guess_history.append(guess_info)
                            words_tried += 1

                            if verbose and score > 20:
                                print(f"{'  ' * depth}Guess {guess_info['guess_number']}: {word} -> {score:.2f}")

                            if score == 100.0:
                                if verbose:
                                    print(f"\n✅ Found target word in {guess_info['guess_number']} guesses!")
                                return True

                            # If we find score > 23, immediately go deeper
                            if score > 23:
                                if verbose:
                                    print(
                                        f"\n{'  ' * depth}💡 Found promising word with score {score:.2f}! Diving deeper...")

                                found = recursive_cluster_search(
                                    cluster_words,
                                    cluster_vectors,
                                    depth + 1,
                                    f"{parent_cluster}{cluster_id}-" if parent_cluster else f"{cluster_id}-"
                                )

                                if found:
                                    return True
                                break  # Don't try more words from this cluster

                            # Limit words tried per cluster
                            if words_tried >= 10:
                                break

            return False

        # Start the recursive search
        found = recursive_cluster_search(vocabulary, word_vectors, depth=0)

        # Create and return summary
        return self._create_solution_summary()

    def _create_solution_summary(self) -> Dict:
        """Create a summary of the solution attempt."""
        best_guess = max(self.guess_history, key=lambda x: x['score'])

        # Count guesses by type
        guess_types = defaultdict(int)
        for guess in self.guess_history:
            guess_types[guess['type']] += 1

        return {
            'target_word': self.target_word,
            'found': any(g['score'] == 100.0 for g in self.guess_history),
            'total_guesses': len(self.guess_history),
            'best_guess': best_guess,
            'guess_types': dict(guess_types),
            'guess_history': self.guess_history
        }

    def run_batch_experiment(self, target_words: List[str], n_clusters: int = 10,
                             save_results: bool = True) -> pd.DataFrame:
        """
        Run automated solving on multiple target words.

        Args:
            target_words: List of target words
            n_clusters: Number of clusters to use
            save_results: Whether to save results to CSV

        Returns:
            DataFrame with experiment results
        """
        results = []

        for target in tqdm(target_words, desc="Solving words"):
            solution = self.solve_with_clustering(target, n_clusters, visualize=False, verbose=False)

            if solution:
                results.append({
                    'target_word': target,
                    'found': solution['found'],
                    'total_guesses': solution['total_guesses'],
                    'best_score': solution['best_guess']['score'],
                    'best_word': solution['best_guess']['word'],
                    'medoid_guesses': solution['guess_types'].get('medoid', 0),
                    'sub_medoid_guesses': solution['guess_types'].get('sub-medoid', 0),
                    'cluster_member_guesses': solution['guess_types'].get('cluster-member', 0)
                })

        df = pd.DataFrame(results)

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantle_clustering_results_{timestamp}.csv"
            filepath = os.path.join(self.results_folder, filename)
            df.to_csv(filepath, index=False)
            print(f"\n📊 Results saved to {filepath}")

        # Print statistics
        print("\n📈 Experiment Statistics:")
        print(f"Total words tested: {len(target_words)}")
        print(f"Success rate: {df['found'].mean() * 100:.1f}%")
        print(f"Average guesses (all): {df['total_guesses'].mean():.2f}")

        if df['found'].any():
            successful_df = df[df['found']]
            print(f"Average guesses (successful only): {successful_df['total_guesses'].mean():.2f}")

        return df

    def analyze_cluster_performance(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze which clustering strategies worked best.

        Args:
            results_df: DataFrame with experiment results

        Returns:
            Dictionary with analysis
        """
        analysis = {
            'total_experiments': len(results_df),
            'success_rate': results_df['found'].mean(),
            'avg_guesses_all': results_df['total_guesses'].mean(),
            'avg_medoid_guesses': results_df['medoid_guesses'].mean(),
            'avg_sub_medoid_guesses': results_df['sub_medoid_guesses'].mean(),
            'avg_cluster_member_guesses': results_df['cluster_member_guesses'].mean()
        }

        # Successful cases only
        successful = results_df[results_df['found']]
        if len(successful) > 0:
            analysis['avg_guesses_successful'] = successful['total_guesses'].mean()
            analysis['median_guesses_successful'] = successful['total_guesses'].median()

        return analysis


# Example usage
if __name__ == "__main__":
    # Configuration
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"

    # Create solver
    solver = AutomatedSemantleSolver(word2vec_path, vocabulary_path)

    # Example 1: Solve single word with detailed output
    print("=== Single Word Example ===")
    solution = solver.solve_with_clustering("book", n_clusters=10, verbose=True)

    # Example 2: Run batch experiment
    # print("\n=== Batch Experiment ===")
    #
    # # Get random sample of words
    # vocabulary = solver.clustering_system.loader.get_vocabulary()
    # test_words = random.sample(vocabulary, 50)  # Start with 50 words
    #
    # # Run experiment
    # results_df = solver.run_batch_experiment(test_words, n_clusters=10)
    #
    # # Analyze results
    # analysis = solver.analyze_cluster_performance(results_df)
    # print("\n📊 Performance Analysis:")
    # for key, value in analysis.items():
    #     print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    #
    # # Example 3: Test different numbers of clusters
    # print("\n=== Testing Different Cluster Numbers ===")
    # cluster_numbers = [5, 10, 15, 20]
    # comparison_results = []
    #
    # for n in cluster_numbers:
    #     print(f"\nTesting with {n} clusters...")
    #     df = solver.run_batch_experiment(test_words[:20], n_clusters=n, save_results=False)
    #
    #     comparison_results.append({
    #         'n_clusters': n,
    #         'success_rate': df['found'].mean(),
    #         'avg_guesses': df['total_guesses'].mean()
    #     })
    #
    # comparison_df = pd.DataFrame(comparison_results)
    # print("\n📊 Cluster Number Comparison:")
    # print(comparison_df)
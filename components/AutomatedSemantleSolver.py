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
try:
    from .VocabularyClusteringSystem import VocabularyClusteringSystem
    from .SemanticExplorer import SemanticExplorer
    from .ClusterAnalyzer import ClusterAnalyzer
    from .KMeansClusterer import KMeansClusterer
    from .Visualizer import Visualizer
    from .DataExporter import DataExporter
except ImportError:
    # Fallback for direct execution
    from VocabularyClusteringSystem import VocabularyClusteringSystem
    from SemanticExplorer import SemanticExplorer
    from ClusterAnalyzer import ClusterAnalyzer
    from KMeansClusterer import KMeansClusterer
    from Visualizer import Visualizer
    from DataExporter import DataExporter


class AutomatedSemantleSolver:
    """
    Automated Semantle solver using clustering strategy.
    Combines the simulator with clustering system for intelligent guessing.
    """
    
    # Search limits to prevent infinite loops and excessive resource usage
    MAX_RECURSION_DEPTH = 6
    MAX_TOTAL_GUESSES = 200

    def __init__(self, word2vec_path: str, vocabulary_path: str, base_results_folder: str = None):
        """
        Initialize the automated solver.

        Args:
            word2vec_path: Path to Word2Vec model
            vocabulary_path: Path to vocabulary file
            base_results_folder: Base folder for test results (default: Tests folder)
        """
        # Create timestamped folder for this run
        if base_results_folder is None:
            base_results_folder = "Tests"

        # Create clean timestamp: YYYYMMDD_HHMM
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_name = f"run_{timestamp}"
        self.results_folder = os.path.join(base_results_folder, self.run_name)

        # Create the directories
        os.makedirs(self.results_folder, exist_ok=True)

        print(f"Created test folder: {self.run_name}")

        # Initialize clustering system with the new folder
        self.clustering_system = VocabularyClusteringSystem(
            word2vec_path=word2vec_path,
            vocabulary_path=vocabulary_path,
            results_folder=self.results_folder,
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
        self.cluster_dive_count = 0  # Track cluster dives for visualization
        
        # Initialize visualizer and exporter
        self.visualizer = Visualizer(self.results_folder)
        self.exporter = DataExporter(self.results_folder)

        # Create summary file for this run
        self.summary_file = os.path.join(self.results_folder, "run_summary.txt")
        with open(self.summary_file, 'w') as f:
            f.write(f"Semantle Clustering Experiment\n")
            f.write(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")

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
                              visualize: bool = True, verbose: bool = True, 
                              use_smart_medoids: bool = True) -> Dict:
        """
        Solve Semantle using clustering strategy.

        Args:
            target_word: The target word to find
            n_clusters: Number of initial clusters
            visualize: Whether to create visualizations
            verbose: Whether to print progress
            use_smart_medoids: Whether to use predefined smart medoids for initial clustering

        Returns:
            Dictionary with solution details
        """
        self.target_word = target_word
        self.guess_history = []
        self.cluster_exploration = defaultdict(list)
        self.cluster_dive_count = 0

        if verbose:
            print(f"\nTarget word: {target_word}")
            if use_smart_medoids:
                print("Using smart medoids for initial clustering...")
            else:
                print(f"Creating {n_clusters} initial clusters...")

        # Step 1: Load data and create initial clusters
        self.clustering_system.load_data()

        # Start recursive clustering process
        vocabulary = self.clustering_system.loader.get_vocabulary()
        word_vectors = self.clustering_system.loader.get_word_vectors()

        # Check if target word is in vocabulary
        if target_word not in vocabulary:
            print(f"Target word '{target_word}' not in vocabulary!")
            return None

        # Define enhanced smart medoids for initial clustering
        # Original diverse coverage + spatial/positional coverage
        smart_medoids = [
            "person", "house", "water", "think", "red", "big", "computer", "animal", "music", "move",  # Original
            "inside", "outside", "center", "edge", "around"  # Spatial/positional coverage
        ]
        
        # Filter smart medoids to only include those in vocabulary
        available_smart_medoids = [word for word in smart_medoids if word in vocabulary]
        if verbose and use_smart_medoids:
            print(f"Smart medoids available: {available_smart_medoids}")
            if len(available_smart_medoids) < len(smart_medoids):
                missing = set(smart_medoids) - set(available_smart_medoids)
                print(f"Missing from vocabulary: {missing}")

        # Recursive clustering function
        def recursive_cluster_search(words_to_cluster, vectors_to_cluster, depth=0, parent_cluster="", create_viz=False):
            """Recursively search through clusters."""

            # Limit recursion depth to avoid infinite loops and excessive guessing
            if depth > self.MAX_RECURSION_DEPTH or len(words_to_cluster) < 10 or len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                if len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                    if verbose:
                        print(f"{'  ' * depth}Reached maximum guess limit of {self.MAX_TOTAL_GUESSES}. Stopping search.")
                return False

            # Create clusters
            n_sub_clusters = min(10, max(2, len(words_to_cluster) // 10))

            if verbose:
                print(
                    f"\n{'  ' * depth}Level {depth}: Creating {n_sub_clusters} clusters from {len(words_to_cluster)} words...")

            sub_clusterer = KMeansClusterer('cosine')
            sub_clusters = sub_clusterer.fit(vectors_to_cluster, n_sub_clusters)

            # Create analyzer
            sub_analyzer = ClusterAnalyzer(words_to_cluster, vectors_to_cluster, sub_clusters, sub_clusterer)
            medoids = sub_analyzer.get_cluster_medoids()
            
            # Create visualization for this cluster dive if enabled
            if create_viz or visualize:
                self.cluster_dive_count += 1
                cluster_name = parent_cluster if parent_cluster else "root"
                viz_filename = f"cluster_dive_{self.cluster_dive_count:03d}_depth{depth}_{cluster_name}_{target_word}.html"
                
                if verbose:
                    print(f"{'  ' * depth}Creating visualization: {viz_filename}")
                
                try:
                    self.visualizer.plot_clusters(words_to_cluster, vectors_to_cluster, sub_clusters, sub_clusterer, viz_filename)
                    
                    # Also create medoids visualization for this dive
                    medoids_viz_filename = f"medoids_dive_{self.cluster_dive_count:03d}_depth{depth}_{cluster_name}_{target_word}.html"
                    medoids_csv_filename = f"medoids_dive_{self.cluster_dive_count:03d}_depth{depth}_{cluster_name}_{target_word}.csv"
                    self.visualizer.plot_medoids_and_paths_with_distances(
                        words_to_cluster, vectors_to_cluster, sub_clusters, sub_analyzer,
                        medoids_viz_filename, self.exporter, medoids_csv_filename
                    )
                except Exception as viz_error:
                    if verbose:
                        print(f"{'  ' * depth}Visualization error: {viz_error}")

            # ALWAYS test ALL medoids first
            medoid_scores = []
            if verbose:
                print(f"{'  ' * depth}Testing all {len(medoids)} medoids...")

            for cluster_id, medoid_word in medoids.items():
                # Check if we've reached the guess limit
                if len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                    if verbose:
                        print(f"{'  ' * depth}Reached maximum guess limit of {self.MAX_TOTAL_GUESSES}. Stopping medoid testing.")
                    break
                    
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
                        print(f"\nFound target word in {guess_info['guess_number']} guesses!")
                    return True

            # Sort by score to find best medoid
            medoid_scores.sort(key=lambda x: x[2], reverse=True)
            best_cluster_id, best_medoid, best_score = medoid_scores[0]

            if verbose:
                print(f"\n{'  ' * depth}Best medoid: {best_medoid} with score {best_score:.2f}")

            # Now decide what to do based on best score
            if best_score > 60:
                # VERY high score! Focus only on this cluster - no more recursion
                if verbose:
                    print(
                        f"\n{'  ' * depth}EXCELLENT score ({best_score:.2f})! Focusing on Cluster {best_cluster_id} only...")

                # Get words in best cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == best_cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]

                if verbose:
                    print(f"{'  ' * depth}Exhaustively searching {len(cluster_words)} words in this cluster...")

                # Sort by distance to medoid
                words_sorted = sorted(
                    cluster_words,
                    key=lambda w: sub_analyzer.find_distance_to_centroid(w, 'cosine')
                )

                # Try ALL words in this cluster
                for word in words_sorted:
                    if word not in [g['word'] for g in self.guess_history]:
                        # Check if we've reached the guess limit
                        if len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                            if verbose:
                                print(f"{'  ' * depth}Reached maximum guess limit of {self.MAX_TOTAL_GUESSES}. Stopping exhaustive search.")
                            return False
                            
                        score = self.calculate_similarity(word, target_word)

                        guess_info = {
                            'word': word,
                            'score': score,
                            'cluster': f"{parent_cluster}{best_cluster_id}" if parent_cluster else str(best_cluster_id),
                            'guess_number': len(self.guess_history) + 1,
                            'type': f'exhaustive-search-{depth}'
                        }

                        self.guess_history.append(guess_info)

                        if verbose and score > 60:  # Show high scores
                            print(f"{'  ' * depth}Guess {guess_info['guess_number']}: {word} -> {score:.2f}")

                        if score == 100.0:
                            if verbose:
                                print(f"\nFound target word in {guess_info['guess_number']} guesses!")
                            return True

                if verbose:
                    print(f"\n{'  ' * depth}Target not in this cluster. Best was {best_medoid} ({best_score:.2f})")
                return False

            elif best_score > 50:
                # Good score - try this cluster first, then maybe go deeper
                if verbose:
                    print(f"\n{'  ' * depth}Good score ({best_score:.2f})! Exploring Cluster {best_cluster_id}...")

                # Get words in best cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == best_cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]
                cluster_vectors = vectors_to_cluster[cluster_indices]

                # First, try some words near the medoid
                words_sorted = sorted(
                    cluster_words,
                    key=lambda w: sub_analyzer.find_distance_to_centroid(w, 'cosine')
                )

                found_better = False
                for word in words_sorted[:min(20, len(words_sorted))]:
                    if word not in [g['word'] for g in self.guess_history]:
                        # Check if we've reached the guess limit
                        if len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                            if verbose:
                                print(f"{'  ' * depth}Reached maximum guess limit of {self.MAX_TOTAL_GUESSES}. Stopping exploration.")
                            return False
                            
                        score = self.calculate_similarity(word, target_word)

                        guess_info = {
                            'word': word,
                            'score': score,
                            'cluster': f"{parent_cluster}{best_cluster_id}" if parent_cluster else str(best_cluster_id),
                            'guess_number': len(self.guess_history) + 1,
                            'type': f'exploration-{depth}'
                        }

                        self.guess_history.append(guess_info)

                        if verbose and score > 40:
                            print(f"{'  ' * depth}Guess {guess_info['guess_number']}: {word} -> {score:.2f}")

                        if score == 100.0:
                            if verbose:
                                print(f"\nFound target word in {guess_info['guess_number']} guesses!")
                            return True

                        if score > best_score + 10:  # Found much better word
                            found_better = True
                            break

                # Only go deeper if cluster is large enough and we found improvement
                if len(cluster_words) >= 50 and depth < 2 and (found_better or best_score > 60):
                    found = recursive_cluster_search(
                        cluster_words,
                        cluster_vectors,
                        depth + 1,
                        f"{parent_cluster}{best_cluster_id}-" if parent_cluster else f"{best_cluster_id}-",
                        create_viz=True
                    )

                    if found:
                        return True

            elif best_score > 23:
                # Moderate score - limited exploration
                if verbose:
                    print(
                        f"{'  ' * depth}Moderate score ({best_score:.2f}). Limited exploration of Cluster {best_cluster_id}...")

                # Get words in best cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == best_cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]
                cluster_vectors = vectors_to_cluster[cluster_indices]

                # Only go deeper if we have many words and shallow depth
                if len(cluster_words) >= 100 and depth == 0:
                    found = recursive_cluster_search(
                        cluster_words,
                        cluster_vectors,
                        depth + 1,
                        f"{parent_cluster}{best_cluster_id}-" if parent_cluster else f"{best_cluster_id}-",
                        create_viz=True
                    )

                    if found:
                        return True

            # If no medoid > 23, try exploring clusters in order of best scores
            if verbose:
                print(f"\n{'  ' * depth}No medoid > 23. Exploring clusters by best score...")

            # Try top 3 clusters
            for cluster_id, medoid_word, medoid_score in medoid_scores[:3]:
                if medoid_score < 5:  # Skip very low scoring clusters
                    continue

                if verbose:
                    print(f"\n{'  ' * depth}Exploring Cluster {cluster_id} (medoid score: {medoid_score:.2f})")

                # Get words in this cluster
                cluster_indices = [i for i, c in enumerate(sub_clusters) if c == cluster_id]
                cluster_words = [words_to_cluster[i] for i in cluster_indices]
                cluster_vectors = vectors_to_cluster[cluster_indices]

                # If cluster is large enough AND we haven't already processed it, do recursive clustering
                cluster_path = f"{parent_cluster}{cluster_id}" if parent_cluster else str(cluster_id)

                # Check if we've already deeply explored this cluster path
                already_explored = any(
                    g['cluster'].startswith(cluster_path + '-')
                    for g in self.guess_history
                )

                if len(cluster_words) >= 50 and depth < 3 and not already_explored:
                    found = recursive_cluster_search(
                        cluster_words,
                        cluster_vectors,
                        depth + 1,
                        f"{cluster_path}-",
                        create_viz=True
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
                            # Check if we've reached the guess limit
                            if len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                                if verbose:
                                    print(f"{'  ' * depth}Reached maximum guess limit of {self.MAX_TOTAL_GUESSES}. Stopping cluster exploration.")
                                return False
                                
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
                                    print(f"\nFound target word in {guess_info['guess_number']} guesses!")
                                return True

                            # If we find score > 23, immediately go deeper
                            if score > 23:
                                if verbose:
                                    print(
                                        f"\n{'  ' * depth}Found promising word with score {score:.2f}! Diving deeper...")

                                found = recursive_cluster_search(
                                    cluster_words,
                                    cluster_vectors,
                                    depth + 1,
                                    f"{parent_cluster}{cluster_id}-" if parent_cluster else f"{cluster_id}-",
                                    create_viz=True
                                )

                                if found:
                                    return True
                                break  # Don't try more words from this cluster

                            # Limit words tried per cluster
                            if words_tried >= 10:
                                break

            return False

        # Start the search process
        if use_smart_medoids and available_smart_medoids:
            # Use smart medoids approach
            if verbose:
                print(f"Testing {len(available_smart_medoids)} smart medoids...")
            
            best_smart_score = 0
            best_smart_word = None
            
            # Test all smart medoids first
            for i, medoid in enumerate(available_smart_medoids):
                # Check if we've reached the guess limit
                if len(self.guess_history) >= self.MAX_TOTAL_GUESSES:
                    if verbose:
                        print(f"Reached maximum guess limit of {self.MAX_TOTAL_GUESSES}. Stopping smart medoid testing.")
                    break
                    
                score = self.calculate_similarity(medoid, target_word)
                
                guess_info = {
                    'word': medoid,
                    'score': score,
                    'cluster': f"Smart-{i}",
                    'guess_number': len(self.guess_history) + 1,
                    'type': 'smart_medoid'
                }
                
                self.guess_history.append(guess_info)
                
                if verbose:
                    print(f"Guess {guess_info['guess_number']}: {medoid} (Smart Medoid {i}) -> {score:.2f}")
                
                if score == 100.0:
                    if verbose:
                        print(f"\nFound target word in {guess_info['guess_number']} guesses!")
                    found = True
                    return self._create_solution_summary()
                
                if score > best_smart_score:
                    best_smart_score = score
                    best_smart_word = medoid
            
            if verbose:
                print(f"\nBest smart medoid: {best_smart_word} with score {best_smart_score:.2f}")
            
            # Now use the best smart medoid to guide further clustering
            # Find words similar to the best smart medoid and cluster around it
            if best_smart_score > 20:  # Only proceed if we have a reasonable score
                if verbose:
                    print(f"Exploring neighborhood around '{best_smart_word}'...")
                
                # Get top 1000 words most similar to best smart medoid
                best_medoid_vector = self.clustering_system.loader.wv[best_smart_word].reshape(1, -1)
                similarities = []
                
                for word in vocabulary:
                    if word != best_smart_word:
                        try:
                            sim = 1 - spatial.distance.cosine(
                                self.clustering_system.loader.wv[best_smart_word],
                                self.clustering_system.loader.wv[word]
                            )
                            similarities.append((word, sim))
                        except:
                            continue
                
                # Sort by similarity and take top words
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_words = [word for word, sim in similarities[:1000]]
                top_vectors = np.array([self.clustering_system.loader.wv[word] for word in top_words])
                
                # Now do recursive clustering on this focused subset
                found = recursive_cluster_search(top_words, top_vectors, depth=0, parent_cluster=f"Smart-{best_smart_word}-", create_viz=True)
            else:
                # If no good smart medoid, fall back to regular clustering
                found = recursive_cluster_search(vocabulary, word_vectors, depth=0, create_viz=True)
        else:
            # Fall back to regular clustering
            found = recursive_cluster_search(vocabulary, word_vectors, depth=0, create_viz=True)

        # Create visualization if requested
        if visualize:
            self._create_visualizations(vocabulary, word_vectors)
        
        # Create and return summary
        return self._create_solution_summary()

    def _create_solution_summary(self) -> Dict:
        """Create a summary of the solution attempt."""
        best_guess = max(self.guess_history, key=lambda x: x['score'])

        # Count guesses by type
        guess_types = defaultdict(int)
        for guess in self.guess_history:
            guess_types[guess['type']] += 1

        summary = {
            'target_word': self.target_word,
            'found': any(g['score'] == 100.0 for g in self.guess_history),
            'total_guesses': len(self.guess_history),
            'best_guess': best_guess,
            'guess_types': dict(guess_types),
            'guess_history': self.guess_history
        }

        # Write to summary file
        with open(self.summary_file, 'a') as f:
            f.write(f"\nTarget: {self.target_word}\n")
            f.write(f"Found: {'Yes' if summary['found'] else 'No'}\n")
            f.write(f"Total guesses: {summary['total_guesses']}\n")
            f.write(f"Best guess: {best_guess['word']} ({best_guess['score']:.2f})\n")
            f.write("-" * 30 + "\n")

        return summary
    
    def _create_visualizations(self, vocabulary, word_vectors):
        """Create HTML visualizations for the clustering results."""
        try:
            print(f"\nCreating visualizations...")
            
            # Create initial clustering for visualization
            clusterer = KMeansClusterer('cosine')
            clusters = clusterer.fit(word_vectors, 10)
            
            # Create analyzer
            analyzer = ClusterAnalyzer(vocabulary, word_vectors, clusters, clusterer)
            
            # Generate cluster visualization
            viz_filename = f"clusters_visualization_{self.target_word}.html"
            self.visualizer.plot_clusters(vocabulary, word_vectors, clusters, clusterer, viz_filename)
            
            # Generate medoids visualization with paths
            medoids_viz_filename = f"medoids_paths_{self.target_word}.html"
            medoids_csv_filename = f"medoids_distances_{self.target_word}.csv"
            self.visualizer.plot_medoids_and_paths_with_distances(
                vocabulary, word_vectors, clusters, analyzer,
                medoids_viz_filename, self.exporter, medoids_csv_filename
            )
            
            # Create summary of guess history CSV
            guess_history_filename = f"guess_history_{self.target_word}.csv"
            guess_df = pd.DataFrame(self.guess_history)
            guess_csv_path = os.path.join(self.results_folder, guess_history_filename)
            guess_df.to_csv(guess_csv_path, index=False)
            print(f"Guess history saved to {guess_history_filename}")
            
            print(f"Visualizations created in folder: {self.run_name}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")

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

        print(f"\nStarting batch experiment with {len(target_words)} words...")
        print(f"Results will be saved to: {self.run_name}\n")

        for i, target in enumerate(tqdm(target_words, desc="Solving words")):
            print(f"\n{'=' * 60}")
            print(f"Word {i + 1}/{len(target_words)}: {target}")
            print(f"{'=' * 60}")

            solution = self.solve_with_clustering(target, n_clusters, visualize=False, verbose=True)

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
            # Save detailed results
            filename = f"clustering_results_{n_clusters}_clusters.csv"
            filepath = os.path.join(self.results_folder, filename)
            df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filename}")

            # Save summary statistics
            stats_file = os.path.join(self.results_folder, "experiment_statistics.txt")
            with open(stats_file, 'w') as f:
                f.write(f"Experiment Statistics\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Total words tested: {len(target_words)}\n")
                f.write(f"Success rate: {df['found'].mean() * 100:.1f}%\n")
                f.write(f"Average guesses (all): {df['total_guesses'].mean():.2f}\n")

                if df['found'].any():
                    successful_df = df[df['found']]
                    f.write(f"Average guesses (successful only): {successful_df['total_guesses'].mean():.2f}\n")
                    f.write(f"Median guesses (successful only): {successful_df['total_guesses'].median():.0f}\n")

                f.write(f"\nFailed words:\n")
                failed = df[~df['found']]
                for _, row in failed.iterrows():
                    f.write(f"- {row['target_word']} (best: {row['best_word']} @ {row['best_score']:.2f})\n")

        # Print statistics
        print("\nExperiment Statistics:")
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
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Automated Semantle Solver with Clustering')
    parser.add_argument('target_word', nargs='?', help='The target word to solve for')
    parser.add_argument('--medoids', choices=['smart', 'random'], default='smart',
                        help='Choose between smart medoids (default) or random medoids for initial clustering')
    parser.add_argument('--clusters', type=int, default=10,
                        help='Number of initial clusters (default: 10, used with random medoids)')
    parser.add_argument('--no-viz', action='store_true', 
                        help='Disable visualizations for faster execution')
    
    args = parser.parse_args()
    
    # Configuration
    word2vec_path = "/mnt/c/Users/User/Downloads/final project/final project/FinalProject/MyData/GoogleNews-vectors-negative300.bin"
    vocabulary_path = "/mnt/c/Users/User/Downloads/final project/final project/FinalProject/MyData/English-Words_Semantle_filtered.txt"

    # Get target word
    if args.target_word:
        target_word = args.target_word
    else:
        target_word = input("Enter the target word: ")
    
    # Determine medoid strategy
    use_smart_medoids = args.medoids == 'smart'
    
    print(f"Target word: {target_word}")
    print(f"Medoid strategy: {'Smart medoids' if use_smart_medoids else 'Random medoids'}")
    if not use_smart_medoids:
        print(f"Number of clusters: {args.clusters}")

    # Create solver
    solver = AutomatedSemantleSolver(word2vec_path, vocabulary_path)

    # Solve the word with visualization
    solution = solver.solve_with_clustering(
        target_word, 
        n_clusters=args.clusters, 
        verbose=True, 
        use_smart_medoids=use_smart_medoids, 
        visualize=not args.no_viz
    )
    
    if solution:
        print(f"\nSolution completed!")
        print(f"Found: {'Yes' if solution['found'] else 'No'}")
        print(f"Total guesses: {solution['total_guesses']}")
        print(f"Best guess: {solution['best_guess']['word']} ({solution['best_guess']['score']:.2f})")
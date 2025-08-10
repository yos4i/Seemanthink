import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import os


class SemantleAutomation:
    """Automated system for solving Semantle puzzles using clustering strategies."""

    def __init__(self, semantic_explorer, analyzer, word_vectors, vocabulary):
        self.explorer = semantic_explorer
        self.analyzer = analyzer
        self.word_vectors = word_vectors
        self.vocabulary = vocabulary
        self.results_folder = "automation_results"
        os.makedirs(self.results_folder, exist_ok=True)

    def simulate_semantle_score(self, guess_word: str, target_word: str) -> float:
        """
        Simulate Semantle score based on cosine similarity.
        Semantle uses cosine similarity * 100 for scoring.
        """
        if guess_word not in self.vocabulary or target_word not in self.vocabulary:
            return -100.0

        guess_vec = self.word_vectors[self.vocabulary.index(guess_word)]
        target_vec = self.word_vectors[self.vocabulary.index(target_word)]

        similarity = cosine_similarity([guess_vec], [target_vec])[0][0]
        return similarity * 100

    def calculate_cluster_entry_threshold(self, medoid_scores: Dict[int, float],
                                          target_cluster: int) -> float:
        """
        Dynamic threshold calculation based on medoid scores distribution.

        Strategy:
        1. If target cluster's medoid has high score, lower threshold
        2. If other medoids have similar scores, raise threshold
        3. Consider score gradient between clusters
        """
        target_score = medoid_scores.get(target_cluster, 0)
        other_scores = [s for c, s in medoid_scores.items() if c != target_cluster]

        if not other_scores:
            return 25.0  # Default threshold

        # Calculate statistics
        avg_other_scores = np.mean(other_scores)
        max_other_score = max(other_scores)
        score_gap = target_score - max_other_score

        # Dynamic threshold based on score distribution
        if score_gap > 20:  # Clear winner
            threshold = min(target_score - 10, 20)
        elif score_gap > 10:  # Good lead
            threshold = min(target_score - 5, 25)
        elif score_gap > 5:  # Small lead
            threshold = min(target_score, 30)
        else:  # Competitive clusters
            threshold = target_score + 5

        # Ensure reasonable bounds
        threshold = max(15, min(threshold, 40))

        return threshold

    def adaptive_medoid_selection(self, medoids: Dict[int, str],
                                  guess_history: List[Dict],
                                  exploration_strategy: str = "balanced") -> str:
        """
        Advanced medoid selection strategies.

        Strategies:
        - 'balanced': Balance between exploration and exploitation
        - 'exploit': Focus on high-scoring areas
        - 'explore': Maximize coverage of search space
        - 'gradient': Follow score gradients
        """
        if not guess_history:
            # Start with first medoid
            return list(medoids.values())[0]

        # Get unvisited medoids
        visited = [g['word'] for g in guess_history]
        unvisited_medoids = [(cid, w) for cid, w in medoids.items() if w not in visited]

        if not unvisited_medoids:
            return None

        # If only one unvisited, return it
        if len(unvisited_medoids) == 1:
            return unvisited_medoids[0][1]

        last_guess = guess_history[-1]
        last_vec = self.word_vectors[self.vocabulary.index(last_guess['word'])]

        if exploration_strategy == "exploit":
            # Choose medoid closest to highest scoring guess
            best_guess = max(guess_history, key=lambda x: x['score'])
            best_vec = self.word_vectors[self.vocabulary.index(best_guess['word'])]

            distances = []
            for cid, word in unvisited_medoids:
                vec = self.word_vectors[self.vocabulary.index(word)]
                dist = euclidean_distances([best_vec], [vec])[0][0]
                distances.append((word, dist))

            return min(distances, key=lambda x: x[1])[0]

        elif exploration_strategy == "explore":
            # Choose medoid farthest from all visited
            max_min_dist = -1
            best_word = unvisited_medoids[0][1]

            for cid, word in unvisited_medoids:
                vec = self.word_vectors[self.vocabulary.index(word)]
                min_dist = float('inf')

                for visited_word in visited:
                    visited_vec = self.word_vectors[self.vocabulary.index(visited_word)]
                    dist = euclidean_distances([vec], [visited_vec])[0][0]
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_word = word

            return best_word

        elif exploration_strategy == "gradient":
            # Follow score improvement gradient
            if len(guess_history) < 2:
                # Not enough history, use explore
                return self.adaptive_medoid_selection(medoids, guess_history, "explore")

            # Calculate score gradient direction
            recent_guesses = guess_history[-3:] if len(guess_history) >= 3 else guess_history
            score_diff = recent_guesses[-1]['score'] - recent_guesses[0]['score']

            if score_diff > 0:  # Improving
                # Continue in similar direction (exploit)
                return self.adaptive_medoid_selection(medoids, guess_history, "exploit")
            else:  # Not improving
                # Try different area (explore)
                return self.adaptive_medoid_selection(medoids, guess_history, "explore")

        else:  # balanced (default)
            # Return next unvisited medoid in order
            return unvisited_medoids[0][1]

    def solve_single_target(self, target_word: str,
                            exploration_strategy: str = "balanced",
                            max_guesses: int = 200) -> Dict:
        """
        Automatically solve for a single target word.
        Returns detailed statistics about the solving process.
        """
        if target_word not in self.vocabulary:
            return {
                'target': target_word,
                'found': False,
                'error': 'Target word not in vocabulary'
            }

        # Initialize
        guess_history = []
        medoids = self.analyzer.get_cluster_medoids()
        target_cluster = self.analyzer.find_word_cluster(target_word)
        found = False
        recursive_depth = 0

        # Track performance metrics
        medoid_scores = {}
        cluster_visits = {c: 0 for c in medoids.keys()}

        # Phase 1: Explore all medoids
        print(f"\nPhase 1: Exploring medoids for '{target_word}'...")
        for cluster_id, medoid_word in medoids.items():
            if len(guess_history) >= max_guesses:
                break

            # Make guess
            score = self.simulate_semantle_score(medoid_word, target_word)

            guess_entry = {
                'word': medoid_word,
                'score': score,
                'cluster': cluster_id,
                'guess_number': len(guess_history) + 1
            }
            guess_history.append(guess_entry)
            medoid_scores[cluster_id] = score
            cluster_visits[cluster_id] += 1

            print(f"  Guess {len(guess_history)}: {medoid_word} (C{cluster_id}) -> {score:.2f}")

            # Check if found
            if medoid_word == target_word or score >= 99.9:
                found = True
                print(f"  ✓ Found target!")
                break

        # Phase 2: Deep dive into promising cluster
        if not found and len(guess_history) < max_guesses:
            # Calculate threshold and find best cluster
            threshold = self.calculate_cluster_entry_threshold(
                medoid_scores, target_cluster
            )

            best_cluster = max(medoid_scores.items(), key=lambda x: x[1])[0]
            best_score = medoid_scores[best_cluster]

            print(f"\nPhase 2: Best cluster is {best_cluster} with score {best_score:.2f}")
            print(f"  Threshold: {threshold:.2f}")
            print(f"  Target cluster: {target_cluster}")

            if best_score >= threshold:
                print(f"  Entering cluster {best_cluster} for deep search...")

                # Deep dive into promising cluster
                result = self._recursive_cluster_search(
                    best_cluster, target_word, guess_history, max_guesses
                )
                guess_history.extend(result['guesses'])
                found = result['found']
                recursive_depth = result['depth']
            else:
                print(f"  Score too low to enter any cluster (best: {best_score:.2f} < {threshold:.2f})")

                # Try exploring more with different strategy
                remaining_guesses = max_guesses - len(guess_history)
                if remaining_guesses > 10:
                    print(f"  Trying additional exploration with {remaining_guesses} guesses...")

                    # Get words from top scoring clusters
                    sorted_clusters = sorted(medoid_scores.items(), key=lambda x: x[1], reverse=True)

                    for cluster_id, _ in sorted_clusters[:3]:  # Top 3 clusters
                        if len(guess_history) >= max_guesses:
                            break

                        # Get some words from this cluster
                        cluster_words = [w for w in self.vocabulary
                                         if self.analyzer.find_word_cluster(w) == cluster_id]

                        # Sample a few words
                        sample_size = min(5, len(cluster_words), remaining_guesses)
                        if sample_size > 0:
                            sampled = np.random.choice(cluster_words, size=sample_size, replace=False)

                            for word in sampled:
                                if word in [g['word'] for g in guess_history]:
                                    continue

                                score = self.simulate_semantle_score(word, target_word)
                                guess_entry = {
                                    'word': word,
                                    'score': score,
                                    'cluster': cluster_id,
                                    'guess_number': len(guess_history) + 1
                                }
                                guess_history.append(guess_entry)

                                if word == target_word or score >= 99.9:
                                    found = True
                                    print(f"  ✓ Found target in extended search!")
                                    break

                            if found:
                                break

        # Compile results
        results = {
            'target': target_word,
            'target_cluster': target_cluster,
            'found': found,
            'total_guesses': len(guess_history),
            'guess_history': guess_history,
            'medoid_scores': medoid_scores,
            'cluster_visits': cluster_visits,
            'recursive_depth': recursive_depth,
            'exploration_strategy': exploration_strategy,
            'final_score': guess_history[-1]['score'] if guess_history else 0,
            'avg_score': np.mean([g['score'] for g in guess_history]) if guess_history else 0,
            'max_score': max([g['score'] for g in guess_history]) if guess_history else 0
        }

        return results

    def _recursive_cluster_search(self, cluster_id: int, target_word: str,
                                  parent_history: List[Dict],
                                  max_guesses: int) -> Dict:
        """
        Recursive search within a specific cluster.
        """
        # Get words in this cluster
        cluster_words = [w for w in self.vocabulary
                         if self.analyzer.find_word_cluster(w) == cluster_id]

        print(f"\n  Searching within cluster {cluster_id} ({len(cluster_words)} words)...")

        if len(cluster_words) < 10:
            # Too small to sub-cluster, do exhaustive search
            print(f"  Small cluster - checking all words...")
            sub_history = []
            for word in cluster_words:
                if len(parent_history) + len(sub_history) >= max_guesses:
                    break

                # Skip if already guessed
                if word in [g['word'] for g in parent_history]:
                    continue

                score = self.simulate_semantle_score(word, target_word)
                sub_history.append({
                    'word': word,
                    'score': score,
                    'cluster': cluster_id,
                    'guess_number': len(parent_history) + len(sub_history) + 1
                })

                print(f"    Guess {len(parent_history) + len(sub_history)}: {word} -> {score:.2f}")

                if word == target_word or score >= 99.9:
                    print(f"    ✓ Found target!")
                    return {'found': True, 'guesses': sub_history, 'depth': 1}

            return {'found': False, 'guesses': sub_history, 'depth': 1}

        # Large enough to search intelligently
        print(f"  Large cluster - using similarity-based search...")

        # Get best guess so far
        if parent_history:
            best_guess = max(parent_history, key=lambda x: x['score'])
            best_vec = self.word_vectors[self.vocabulary.index(best_guess['word'])]
            print(f"  Using '{best_guess['word']}' (score: {best_guess['score']:.2f}) as reference")
        else:
            # Use cluster medoid as reference
            medoids = self.analyzer.get_cluster_medoids()
            if cluster_id in medoids:
                best_word = medoids[cluster_id]
                best_vec = self.word_vectors[self.vocabulary.index(best_word)]
                print(f"  Using medoid '{best_word}' as reference")
            else:
                # Fallback to random sampling
                sample_size = min(20, len(cluster_words))
                sampled = np.random.choice(cluster_words, size=sample_size, replace=False)
                sub_history = []

                for word in sampled:
                    if len(parent_history) + len(sub_history) >= max_guesses:
                        break

                    if word in [g['word'] for g in parent_history]:
                        continue

                    score = self.simulate_semantle_score(word, target_word)
                    sub_history.append({
                        'word': word,
                        'score': score,
                        'cluster': cluster_id,
                        'guess_number': len(parent_history) + len(sub_history) + 1
                    })

                    if word == target_word or score >= 99.9:
                        return {'found': True, 'guesses': sub_history, 'depth': 1}

                return {'found': False, 'guesses': sub_history, 'depth': 1}

        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate similarities for all words in cluster
        similarities = []
        for word in cluster_words:
            if word in [g['word'] for g in parent_history]:
                continue

            vec = self.word_vectors[self.vocabulary.index(word)]
            sim = cosine_similarity([best_vec], [vec])[0][0]
            similarities.append((word, sim))

        # Sort by similarity and search top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)

        sub_history = []
        search_size = min(30, len(similarities), max_guesses - len(parent_history))

        print(f"  Checking top {search_size} similar words...")

        for word, sim in similarities[:search_size]:
            if len(parent_history) + len(sub_history) >= max_guesses:
                break

            score = self.simulate_semantle_score(word, target_word)
            sub_history.append({
                'word': word,
                'score': score,
                'cluster': cluster_id,
                'guess_number': len(parent_history) + len(sub_history) + 1
            })

            if len(sub_history) <= 5 or score > 50:  # Print first few or high scores
                print(f"    Guess {len(parent_history) + len(sub_history)}: {word} -> {score:.2f}")

            if word == target_word or score >= 99.9:
                print(f"    ✓ Found target!")
                return {'found': True, 'guesses': sub_history, 'depth': 1}

        print(f"  Searched {len(sub_history)} words in cluster")
        return {'found': False, 'guesses': sub_history, 'depth': 1}

    def batch_solve(self, target_words: List[str],
                    exploration_strategy: str = "balanced",
                    save_results: bool = True) -> pd.DataFrame:
        """
        Solve multiple target words and compile statistics.
        """
        all_results = []

        for i, target in enumerate(target_words):
            print(f"Solving {i + 1}/{len(target_words)}: {target}")
            result = self.solve_single_target(target, exploration_strategy)
            all_results.append(result)

        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'target_word': result['target'],
                'target_cluster': result.get('target_cluster', -1),
                'found': result['found'],
                'total_guesses': result.get('total_guesses', 0),
                'final_score': result.get('final_score', 0),
                'avg_score': result.get('avg_score', 0),
                'max_score': result.get('max_score', 0),
                'strategy': result.get('exploration_strategy', ''),
                'recursive_depth': result.get('recursive_depth', 0)
            })

        df_summary = pd.DataFrame(summary_data)

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save summary
            summary_path = os.path.join(self.results_folder,
                                        f"batch_results_summary_{timestamp}.csv")
            df_summary.to_csv(summary_path, index=False)

            # Save detailed results
            detailed_path = os.path.join(self.results_folder,
                                         f"batch_results_detailed_{timestamp}.pkl")
            pd.to_pickle(all_results, detailed_path)

            print(f"Results saved to {self.results_folder}")

        return df_summary

    def analyze_threshold_performance(self, target_words: List[str],
                                      threshold_range: List[float]) -> pd.DataFrame:
        """
        Test different threshold values to find optimal cluster entry point.
        """
        results = []

        for threshold in threshold_range:
            print(f"Testing threshold: {threshold}")

            # Temporarily override the threshold calculation
            original_method = self.calculate_cluster_entry_threshold
            self.calculate_cluster_entry_threshold = lambda *args: threshold

            # Run batch solve
            df_results = self.batch_solve(target_words, save_results=False)

            # Calculate statistics
            success_rate = df_results['found'].mean()
            avg_guesses = df_results[df_results['found']]['total_guesses'].mean()

            results.append({
                'threshold': threshold,
                'success_rate': success_rate,
                'avg_guesses_when_found': avg_guesses,
                'total_targets': len(target_words),
                'targets_found': df_results['found'].sum()
            })

            # Restore original method
            self.calculate_cluster_entry_threshold = original_method

        return pd.DataFrame(results)
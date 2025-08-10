import numpy as np
from scipy import spatial
from gensim.models import KeyedVectors
import os
import pandas as pd
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm


class SemantleSimulator:
    """
    Simulates the Semantle game environment.
    Allows setting a target word and getting scores for guesses.
    """

    def __init__(self, word2vec_path: str, vocabulary_path: str):
        """
        Initialize the simulator with word vectors and vocabulary.

        Args:
            word2vec_path: Path to Word2Vec model
            vocabulary_path: Path to vocabulary file
        """
        print("Loading Word2Vec model...")
        self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        print("Loading vocabulary...")
        with open(vocabulary_path, 'r', encoding='utf-8') as f:
            self.vocabulary = [word.strip() for word in f.readlines()]

        # Filter vocabulary to only include words in Word2Vec model
        self.vocabulary = [word for word in self.vocabulary if word in self.wv]
        print(f"Loaded {len(self.vocabulary)} words")

        self.target_word = None
        self.guess_history = []

    def set_target_word(self, word: str) -> bool:
        """
        Set the target word for the game.

        Args:
            word: The target word

        Returns:
            True if word is valid, False otherwise
        """
        if word not in self.vocabulary:
            print(f"Word '{word}' not in vocabulary!")
            return False

        self.target_word = word
        self.guess_history = []
        return True

    def get_similarity_score(self, guess_word: str) -> Optional[float]:
        """
        Get the similarity score for a guess word (like in Semantle).

        Args:
            guess_word: The word to guess

        Returns:
            Similarity score (0-100) or None if invalid
        """
        if self.target_word is None:
            print("No target word set!")
            return None

        if guess_word not in self.wv:
            print(f"Word '{guess_word}' not in Word2Vec model!")
            return None

        # Calculate cosine similarity (same as Semantle)
        similarity = (1 - spatial.distance.cosine(
            self.wv[guess_word],
            self.wv[self.target_word]
        )) * 100

        # Round to 2 decimal places like Semantle
        return round(similarity, 2)

    def make_guess(self, guess_word: str) -> Dict:
        """
        Make a guess and record it in history.

        Args:
            guess_word: The word to guess

        Returns:
            Dictionary with guess information
        """
        score = self.get_similarity_score(guess_word)

        if score is not None:
            guess_info = {
                'word': guess_word,
                'score': score,
                'guess_number': len(self.guess_history) + 1,
                'timestamp': datetime.now().isoformat()
            }
            self.guess_history.append(guess_info)
            return guess_info

        return None

    def get_game_summary(self) -> Dict:
        """Get summary of current game."""
        if not self.guess_history:
            return {'status': 'No guesses made yet'}

        return {
            'target_word': self.target_word,
            'total_guesses': len(self.guess_history),
            'best_guess': max(self.guess_history, key=lambda x: x['score']),
            'won': any(g['score'] == 100.0 for g in self.guess_history)
        }


class SemantleExperiment:
    """
    Run experiments with clustering methods on Semantle.
    """

    def __init__(self, simulator: SemantleSimulator, clustering_system):
        """
        Initialize experiment runner.

        Args:
            simulator: SemantleSimulator instance
            clustering_system: Your VocabularyClusteringSystem instance
        """
        self.simulator = simulator
        self.clustering_system = clustering_system
        self.results = []

    def run_single_game(self, target_word: str, method: str = 'medoid') -> Dict:
        """
        Run a single game with clustering strategy.

        Args:
            target_word: The target word
            method: Strategy to use ('medoid', 'random', etc.)

        Returns:
            Game results
        """
        if not self.simulator.set_target_word(target_word):
            return None

        print(f"\nTarget word: {target_word}")

        # Get initial clusters
        clusters, _ = self.clustering_system.cluster(n_clusters=10)
        analyzer = self.clustering_system.analyzer

        if method == 'medoid':
            # Get medoids as initial guesses
            medoids = analyzer.get_cluster_medoids()

            # Try medoids in order
            for cluster_id, medoid_word in medoids.items():
                guess_info = self.simulator.make_guess(medoid_word)
                if guess_info:
                    print(f"Guess {guess_info['guess_number']}: {medoid_word} -> {guess_info['score']:.2f}")

                    # If we found it, we're done
                    if guess_info['score'] == 100.0:
                        print(f"✅ Found target word in {guess_info['guess_number']} guesses!")
                        break

                    # If score is high, focus on that cluster
                    if guess_info['score'] > 70:
                        # Get words from the same cluster
                        target_cluster = analyzer.find_word_cluster(medoid_word)
                        cluster_words = [w for w in self.simulator.vocabulary
                                         if analyzer.find_word_cluster(w) == target_cluster]

                        # Sort by distance to medoid
                        cluster_words.sort(key=lambda w: analyzer.find_distance_to_centroid(w))

                        # Try words in the cluster
                        for word in cluster_words[:20]:  # Try top 20 closest
                            if word not in [g['word'] for g in self.simulator.guess_history]:
                                guess_info = self.simulator.make_guess(word)
                                if guess_info:
                                    print(f"Guess {guess_info['guess_number']}: {word} -> {guess_info['score']:.2f}")
                                    if guess_info['score'] == 100.0:
                                        print(f"✅ Found target word in {guess_info['guess_number']} guesses!")
                                        return self.simulator.get_game_summary()

        elif method == 'random':
            # Random baseline
            words_to_try = self.simulator.vocabulary.copy()
            random.shuffle(words_to_try)

            for word in words_to_try:
                guess_info = self.simulator.make_guess(word)
                if guess_info and guess_info['score'] == 100.0:
                    print(f"✅ Found target word in {guess_info['guess_number']} guesses!")
                    break

        return self.simulator.get_game_summary()

    def run_experiment(self, target_words: List[str], method: str = 'medoid') -> pd.DataFrame:
        """
        Run experiment on multiple target words.

        Args:
            target_words: List of target words to test
            method: Strategy to use

        Returns:
            DataFrame with results
        """
        results = []

        for target in tqdm(target_words, desc=f"Running {method} experiments"):
            game_result = self.run_single_game(target, method)

            if game_result and 'total_guesses' in game_result:
                results.append({
                    'target_word': target,
                    'method': method,
                    'total_guesses': game_result['total_guesses'],
                    'won': game_result['won'],
                    'best_score': game_result['best_guess']['score'] if 'best_guess' in game_result else 0
                })

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, filename: str):
        """Save experiment results."""
        output_path = os.path.join(self.clustering_system.exporter.results_folder, filename)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Average guesses: {df['total_guesses'].mean():.2f}")
        print(f"Success rate: {df['won'].mean() * 100:.1f}%")
        print(f"Median guesses: {df['total_guesses'].median():.0f}")


# Example usage
if __name__ == "__main__":
    # Initialize simulator
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"

    simulator = SemantleSimulator(word2vec_path, vocabulary_path)

    # Example 1: Manual play
    print("=== Manual Play Example ===")
    simulator.set_target_word("slice")

    # Make some guesses
    guesses = ["laptop", "technology", "software", "hardware", "slice"]
    for guess in guesses:
        result = simulator.make_guess(guess)
        if result:
            print(f"{guess}: {result['score']:.2f}")

    print("\nGame Summary:")
    print(simulator.get_game_summary())

    # Example 2: Integration with clustering
    print("\n=== Clustering Integration Example ===")
    from VocabularyClusteringSystem import VocabularyClusteringSystem

    # Initialize clustering system
    clustering_system = VocabularyClusteringSystem(
        word2vec_path=word2vec_path,
        vocabulary_path=vocabulary_path,
        results_folder="semantle_experiments",
        algorithm='kmeans',
        distance_metric='cosine',
        normalizer='standard'
    )

    # Load data and create clusters
    clustering_system.load_data()

    # Create experiment runner
    experiment = SemantleExperiment(simulator, clustering_system)

    # Run on a few target words
    test_words = ["apple", "car", "happy", "science", "music"]
    results_df = experiment.run_experiment(test_words, method='medoid')

    # Save results
    experiment.save_results(results_df, "clustering_performance.csv")

    # Example 3: Large scale experiment
    print("\n=== Large Scale Experiment Setup ===")
    print("To run on 1000 words:")
    print("1. Select 1000 random words from vocabulary:")
    print("   random_targets = random.sample(simulator.vocabulary, 1000)")
    print("2. Run experiment:")
    print("   results = experiment.run_experiment(random_targets, method='medoid')")
    print("3. Compare with baseline:")
    print("   baseline_results = experiment.run_experiment(random_targets[:100], method='random')")
#!/usr/bin/env python3
"""
Quick test script to verify the system works.
"""

import os
import sys
from VocabularyClusteringSystem import VocabularyClusteringSystem
from SemantleAutomation import SemantleAutomation


def quick_test():
    # Configuration
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"
    results_folder = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\quick_test_results"

    os.makedirs(results_folder, exist_ok=True)

    print("Loading system...")

    # Initialize system with fewer clusters for testing
    system = VocabularyClusteringSystem(
        word2vec_path=word2vec_path,
        vocabulary_path=vocabulary_path,
        results_folder=results_folder,
        algorithm='kmeans',
        distance_metric='cosine',
        normalizer='standard'
    )

    # Load data
    system.load_data()

    # Try with different number of clusters
    n_clusters = 10  # You can try 5, 10, 15, 20
    print(f"\nClustering with {n_clusters} clusters...")
    clusters, silhouette_score = system.cluster(n_clusters)

    print(f"Clustering complete. Silhouette score: {silhouette_score:.4f}")

    # Create automation
    automation = SemantleAutomation(
        semantic_explorer=None,
        analyzer=system.analyzer,
        word_vectors=system.loader.get_word_vectors(),
        vocabulary=system.loader.get_vocabulary()
    )

    # Test words
    test_words = ["create", "computer", "happy", "build", "make"]

    print("\n" + "=" * 60)
    print("TESTING AUTOMATION")
    print("=" * 60)

    for word in test_words:
        if word not in automation.vocabulary:
            print(f"\n'{word}' not in vocabulary, skipping...")
            continue

        print(f"\n{'=' * 60}")
        print(f"Testing: {word}")
        print(f"{'=' * 60}")

        # Get word info
        cluster = automation.analyzer.find_word_cluster(word)
        print(f"Word cluster: {cluster}")

        # Test with limited guesses
        result = automation.solve_single_target(word, max_guesses=50)

        print(f"\nResult:")
        print(f"  Found: {result['found']}")
        print(f"  Total guesses: {result['total_guesses']}")
        if result['found']:
            print(f"  ✓ SUCCESS in {result['total_guesses']} guesses!")
        else:
            print(f"  ✗ Not found after {result['total_guesses']} guesses")
            print(f"  Best score achieved: {result['max_score']:.2f}")

    # Test similarity between words
    print("\n" + "=" * 60)
    print("TESTING WORD SIMILARITIES")
    print("=" * 60)

    test_pairs = [
        ("create", "make"),
        ("create", "build"),
        ("create", "destroy"),
        ("happy", "sad"),
        ("happy", "joyful"),
    ]

    for word1, word2 in test_pairs:
        if word1 in automation.vocabulary and word2 in automation.vocabulary:
            score = automation.simulate_semantle_score(word1, word2)
            print(f"{word1} <-> {word2}: {score:.2f}")
        else:
            missing = []
            if word1 not in automation.vocabulary:
                missing.append(word1)
            if word2 not in automation.vocabulary:
                missing.append(word2)
            print(f"Missing words: {missing}")


if __name__ == "__main__":
    quick_test()
#!/usr/bin/env python3
"""
Debug script to identify issues with the Semantle automation system.
"""

import numpy as np
from VocabularyClusteringSystem import VocabularyClusteringSystem
from SemantleAutomation import SemantleAutomation
from SemanticExplorer import SemanticExplorer
import os


def debug_basic_functionality(system, target_word="create"):
    """
    Debug basic system functionality step by step.
    """
    print(f"\n{'=' * 60}")
    print(f"DEBUGGING BASIC FUNCTIONALITY FOR: {target_word}")
    print(f"{'=' * 60}\n")

    # 1. Check if word exists in vocabulary
    vocabulary = system.loader.get_vocabulary()
    if target_word not in vocabulary:
        print(f"ERROR: '{target_word}' not in vocabulary!")
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"Sample words: {vocabulary[:10]}")
        return
    else:
        print(f"✓ '{target_word}' found in vocabulary")

    # 2. Check clustering
    cluster = system.analyzer.find_word_cluster(target_word)
    print(f"✓ Target word cluster: {cluster}")

    # 3. Check medoids
    medoids = system.analyzer.get_cluster_medoids()
    print(f"✓ Number of medoids: {len(medoids)}")
    print("\nMedoids by cluster:")
    for cid, word in medoids.items():
        print(f"  Cluster {cid}: {word}")

    # 4. Test similarity calculation
    print("\n" + "-" * 40)
    print("Testing similarity calculations:")

    # Get vectors
    word_vectors = system.loader.get_word_vectors()
    target_idx = vocabulary.index(target_word)
    target_vec = word_vectors[target_idx]

    # Test with a few medoids
    from sklearn.metrics.pairwise import cosine_similarity
    for cid, medoid in list(medoids.items())[:3]:
        medoid_idx = vocabulary.index(medoid)
        medoid_vec = word_vectors[medoid_idx]

        # Calculate similarity
        sim = cosine_similarity([target_vec], [medoid_vec])[0][0]
        score = sim * 100

        print(f"  {medoid} (C{cid}): similarity={sim:.4f}, score={score:.2f}")

    return True


def debug_automation_step_by_step(automation, target_word="create"):
    """
    Debug the automation system step by step.
    """
    print(f"\n{'=' * 60}")
    print(f"DEBUGGING AUTOMATION FOR: {target_word}")
    print(f"{'=' * 60}\n")

    # Check if word exists
    if target_word not in automation.vocabulary:
        print(f"ERROR: '{target_word}' not in automation vocabulary!")
        return

    # Get medoids
    medoids = automation.analyzer.get_cluster_medoids()
    target_cluster = automation.analyzer.find_word_cluster(target_word)

    print(f"Target cluster: {target_cluster}")
    print(f"Target medoid: {medoids.get(target_cluster, 'N/A')}")

    # Simulate first few guesses manually
    print("\nSimulating first medoid guesses:")
    medoid_scores = {}

    for i, (cid, word) in enumerate(medoids.items()):
        if i >= 5:  # Test first 5
            break

        score = automation.simulate_semantle_score(word, target_word)
        medoid_scores[cid] = score
        print(f"  Guess {i + 1}: {word} (C{cid}) -> {score:.2f}")

    # Calculate threshold
    threshold = automation.calculate_cluster_entry_threshold(medoid_scores, target_cluster)
    print(f"\nCalculated threshold: {threshold:.2f}")

    # Check if we would enter any cluster
    if medoid_scores:
        best_cluster = max(medoid_scores.items(), key=lambda x: x[1])[0]
        best_score = medoid_scores[best_cluster]
        print(f"Best scoring cluster: {best_cluster} (score: {best_score:.2f})")
        print(f"Would enter cluster: {'YES' if best_score >= threshold else 'NO'}")

    # Try actual solve with limited guesses
    print("\n" + "-" * 40)
    print("Running limited solve (max 20 guesses):")

    result = automation.solve_single_target(target_word, max_guesses=20)

    print(f"\nResult:")
    print(f"  Found: {result['found']}")
    print(f"  Total guesses: {result['total_guesses']}")
    print(f"  Final score: {result['final_score']:.2f}")

    if result['guess_history']:
        print(f"\nFirst 5 guesses:")
        for guess in result['guess_history'][:5]:
            print(f"  {guess['guess_number']}. {guess['word']} (C{guess['cluster']}) -> {guess['score']:.2f}")

    return result


def check_recursive_search(automation, target_word="create"):
    """
    Check if recursive search within cluster works.
    """
    print(f"\n{'=' * 60}")
    print(f"CHECKING RECURSIVE SEARCH FOR: {target_word}")
    print(f"{'=' * 60}\n")

    target_cluster = automation.analyzer.find_word_cluster(target_word)

    # Get all words in target cluster
    cluster_words = [w for w in automation.vocabulary
                     if automation.analyzer.find_word_cluster(w) == target_cluster]

    print(f"Target cluster {target_cluster} contains {len(cluster_words)} words")

    if target_word in cluster_words:
        print(f"✓ Target word is in its cluster")
    else:
        print(f"ERROR: Target word not in cluster!")
        return

    # Check if recursive search would find it
    if len(cluster_words) < 10:
        print(f"Small cluster - would check all words")
        print(f"Words in cluster: {cluster_words}")
    else:
        print(f"Large cluster - would use similarity search")

        # Find most similar words to target
        from sklearn.metrics.pairwise import cosine_similarity
        target_vec = automation.word_vectors[automation.vocabulary.index(target_word)]

        similarities = []
        for word in cluster_words[:20]:  # Check first 20
            if word == target_word:
                continue
            vec = automation.word_vectors[automation.vocabulary.index(word)]
            sim = cosine_similarity([target_vec], [vec])[0][0]
            similarities.append((word, sim * 100))

        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\nMost similar words in cluster:")
        for word, score in similarities[:5]:
            print(f"  {word}: {score:.2f}")


def test_with_known_good_case(automation):
    """
    Test with a pair of words that should have high similarity.
    """
    print(f"\n{'=' * 60}")
    print("TESTING WITH KNOWN SIMILAR WORDS")
    print(f"{'=' * 60}\n")

    # Find some highly similar words
    test_pairs = [
        ("create", "make"),
        ("create", "build"),
        ("create", "construct"),
        ("computer", "laptop"),
        ("computer", "machine"),
    ]

    for word1, word2 in test_pairs:
        if word1 in automation.vocabulary and word2 in automation.vocabulary:
            score = automation.simulate_semantle_score(word1, word2)
            cluster1 = automation.analyzer.find_word_cluster(word1)
            cluster2 = automation.analyzer.find_word_cluster(word2)

            print(f"{word1} -> {word2}:")
            print(f"  Score: {score:.2f}")
            print(f"  Clusters: {cluster1} -> {cluster2} {'(same)' if cluster1 == cluster2 else '(different)'}")


def main():
    """
    Main debug function.
    """
    # Configuration
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"
    results_folder = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\debug_results"

    os.makedirs(results_folder, exist_ok=True)

    print("Loading clustering system...")

    # Initialize system
    system = VocabularyClusteringSystem(
        word2vec_path=word2vec_path,
        vocabulary_path=vocabulary_path,
        results_folder=results_folder,
        algorithm='kmeans',
        distance_metric='cosine',
        normalizer='standard'
    )

    # Load data and perform initial clustering
    system.load_data()
    clusters, silhouette_score = system.cluster(10)

    print(f"Clustering complete. Silhouette score: {silhouette_score:.4f}")

    # Create automation
    automation = SemantleAutomation(
        semantic_explorer=None,
        analyzer=system.analyzer,
        word_vectors=system.loader.get_word_vectors(),
        vocabulary=system.loader.get_vocabulary()
    )

    # Run debug tests
    test_words = ["create", "paideia", "computer"]

    for word in test_words:
        if word in system.loader.get_vocabulary():
            print(f"\n{'=' * 80}")
            print(f"TESTING: {word}")
            print(f"{'=' * 80}")

            # Basic functionality
            if debug_basic_functionality(system, word):
                # Automation debug
                debug_automation_step_by_step(automation, word)

                # Recursive search check
                check_recursive_search(automation, word)
        else:
            print(f"\nWARNING: '{word}' not in vocabulary!")

    # Test known similar words
    test_with_known_good_case(automation)

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()



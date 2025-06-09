#!/usr/bin/env python3
"""
Script to run automated Semantle solving experiments.
Tests different strategies and threshold values.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your existing modules
from VocabularyClusteringSystem import VocabularyClusteringSystem
from SemantleAutomation import SemantleAutomation


def load_test_words(file_path=None, n_words=100, vocabulary=None):
    """
    Load test words for experiments.

    Args:
        file_path: Path to file with test words (one per line)
        n_words: Number of words to test
        vocabulary: List of available words
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        return words[:n_words]

    # If no file provided, return None to use default selection
    return None


def run_strategy_comparison(automation, test_words,
                            strategies=None):
    """
    Compare different exploration strategies.
    """
    if strategies is None:
        strategies = ["balanced", "exploit", "explore", "gradient"]

    results = {}

    for strategy in strategies:
        print(f"\n{'=' * 50}")
        print(f"Testing strategy: {strategy}")
        print(f"{'=' * 50}")

        df_results = automation.batch_solve(test_words,
                                            exploration_strategy=strategy,
                                            save_results=True)
        results[strategy] = df_results

        # Print summary statistics
        print(f"\nStrategy: {strategy}")
        print(f"Success rate: {df_results['found'].mean():.2%}")
        print(f"Average guesses (when found): {df_results[df_results['found']]['total_guesses'].mean():.1f}")
        print(f"Median guesses (when found): {df_results[df_results['found']]['total_guesses'].median():.1f}")

    return results


def run_threshold_analysis(automation, test_words, threshold_range=None):
    """
    Analyze performance across different threshold values.
    """
    if threshold_range is None:
        threshold_range = list(range(15, 41, 5))

    print(f"\n{'=' * 50}")
    print("Running threshold analysis")
    print(f"{'=' * 50}")

    df_threshold_results = automation.analyze_threshold_performance(
        test_words, threshold_range
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(automation.results_folder,
                               f"threshold_analysis_{timestamp}.csv")
    df_threshold_results.to_csv(output_path, index=False)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate vs threshold
    ax1.plot(df_threshold_results['threshold'],
             df_threshold_results['success_rate'],
             marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Threshold Value')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs Cluster Entry Threshold')
    ax1.grid(True, alpha=0.3)

    # Average guesses vs threshold
    ax2.plot(df_threshold_results['threshold'],
             df_threshold_results['avg_guesses_when_found'],
             marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Threshold Value')
    ax2.set_ylabel('Average Guesses (when found)')
    ax2.set_title('Efficiency vs Cluster Entry Threshold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(automation.results_folder,
                             f"threshold_analysis_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Threshold analysis saved to {output_path}")
    print(f"Plot saved to {plot_path}")

    return df_threshold_results


def generate_performance_report(automation, strategy_results, threshold_results):
    """
    Generate comprehensive performance report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(automation.results_folder,
                               f"performance_report_{timestamp}.txt")

    with open(report_path, 'w') as f:
        f.write("SEMANTLE AUTOMATION PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Strategy comparison
        f.write("STRATEGY COMPARISON\n")
        f.write("-" * 30 + "\n")

        for strategy, df in strategy_results.items():
            success_rate = df['found'].mean()
            avg_guesses = df[df['found']]['total_guesses'].mean()
            median_guesses = df[df['found']]['total_guesses'].median()

            f.write(f"\n{strategy.upper()} Strategy:\n")
            f.write(f"  Success Rate: {success_rate:.2%}\n")
            f.write(f"  Average Guesses: {avg_guesses:.1f}\n")
            f.write(f"  Median Guesses: {median_guesses:.1f}\n")
            f.write(f"  Min Guesses: {df[df['found']]['total_guesses'].min()}\n")
            f.write(f"  Max Guesses: {df[df['found']]['total_guesses'].max()}\n")

        # Threshold analysis
        f.write("\n\nTHRESHOLD ANALYSIS\n")
        f.write("-" * 30 + "\n")

        best_threshold = threshold_results.loc[
            threshold_results['success_rate'].idxmax(), 'threshold'
        ]
        most_efficient = threshold_results.loc[
            threshold_results['avg_guesses_when_found'].idxmin(), 'threshold'
        ]

        f.write(f"\nBest Success Rate Threshold: {best_threshold}\n")
        f.write(f"Most Efficient Threshold: {most_efficient}\n")

        f.write("\nDetailed Results:\n")
        f.write(threshold_results.to_string(index=False))

        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")

        # Find best strategy
        best_strategy = max(strategy_results.items(),
                            key=lambda x: x[1]['found'].mean())[0]

        f.write(f"\n1. Best Overall Strategy: {best_strategy}\n")
        f.write(f"2. Recommended Threshold: {best_threshold} ")
        f.write(f"(balances success rate and efficiency)\n")
        f.write("\n3. For maximum success rate, use dynamic thresholds ")
        f.write("based on score distribution.\n")
        f.write("4. Consider using 'exploit' strategy when initial ")
        f.write("medoid scores are high (>30).\n")
        f.write("5. Use 'explore' strategy when scores are uniformly low (<20).\n")

    print(f"\nPerformance report saved to {report_path}")


def main():
    """
    Main function to run all experiments.
    """
    # Configuration
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"
    results_folder = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\automation_results"

    # Create results folder
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
    dataset_name = os.path.splitext(os.path.basename(vocabulary_path))[0]
    clusters, silhouette_score = system.run_complete_analysis(10, dataset_name)

    print(f"Initial clustering complete. Silhouette score: {silhouette_score:.4f}")

    # Create automation system
    automation = SemantleAutomation(
        semantic_explorer=None,  # Not needed for automation
        analyzer=system.analyzer,
        word_vectors=system.loader.get_word_vectors(),
        vocabulary=system.loader.get_vocabulary()
    )

    # Select test words
    print("\nSelecting test words...")

    # Get all words from vocabulary
    all_words = system.loader.get_vocabulary()

    # Option 1: Use specific test words
    test_words_full = ["create", "computer", "happy", "build", "make",
                       "love", "house", "car", "book", "tree",
                       "water", "fire", "earth", "air", "music"]

    # Filter to only words in vocabulary
    test_words = []
    for word in test_words_full:
        if word in all_words:
            test_words.append(word)
        else:
            print(f"Warning: '{word}' not in vocabulary")

    # Option 2: Add random words if needed
    if len(test_words) < 20:
        remaining = 20 - len(test_words)
        random_words = np.random.choice(
            [w for w in all_words if w not in test_words],
            size=remaining,
            replace=False
        ).tolist()
        test_words.extend(random_words)

    print(f"Selected {len(test_words)} test words")
    print(f"First 5 test words: {test_words[:5]}")

    # Run experiments
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENTS")
    print("=" * 70)

    # 1. Strategy comparison
    strategy_results = run_strategy_comparison(
        automation, test_words[:20]  # Use subset for faster testing
    )

    # 2. Threshold analysis
    threshold_results = run_threshold_analysis(
        automation, test_words[:20],  # Use subset for faster testing
        threshold_range=list(range(15, 41, 5))
    )

    # 3. Generate report
    generate_performance_report(automation, strategy_results, threshold_results)

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {results_folder}")

    # Quick summary
    print("\nQUICK SUMMARY:")
    for strategy, df in strategy_results.items():
        print(f"{strategy}: {df['found'].mean():.2%} success, "
              f"{df[df['found']]['total_guesses'].mean():.1f} avg guesses")


if __name__ == "__main__":
    main()
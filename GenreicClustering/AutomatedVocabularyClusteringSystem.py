from datetime import datetime
from VocabularyLoader import VocabularyLoader
from DataExporter import DataExporter
from Visualizer import Visualizer
from KMeansClusterer import KMeansClusterer
from ClusterAnalyzer import ClusterAnalyzer
from AutomatedSemanticExplorer import AutomatedSemanticExplorer
import os
import random
import pandas as pd


class AutomatedVocabularyClusteringSystem:
    """
    Extended VocabularyClusteringSystem with automated Semantle simulation.
    """

    def __init__(self, word2vec_path, vocabulary_path, results_folder,
                 algorithm='kmeans', distance_metric='cosine', normalizer=None):
        """Initialize the automated clustering system."""
        self.loader = VocabularyLoader(word2vec_path, vocabulary_path, normalizer)
        self.exporter = DataExporter(results_folder)
        self.visualizer = Visualizer(results_folder)
        self.algorithm_name = algorithm.lower()
        self.distance_metric = distance_metric
        self.algorithm = None
        self.analyzer = None
        self.results_folder = results_folder

        # Initialize clustering algorithm
        if self.algorithm_name == 'kmeans':
            self.algorithm = KMeansClusterer(distance_metric)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def load_data(self):
        """Load vocabulary and word vectors."""
        self.loader.load_vocabulary()

    def cluster(self, n_clusters):
        """Perform clustering."""
        vocabulary = self.loader.get_vocabulary()
        word_vectors = self.loader.get_word_vectors()

        print(f"Testing {self.algorithm_name.upper()} with n_clusters={n_clusters}")
        clusters = self.algorithm.fit(word_vectors, n_clusters)

        # Calculate silhouette score
        silhouette_avg = self.algorithm.calculate_silhouette_score(word_vectors)
        print(f"Silhouette Score ({self.distance_metric}): {silhouette_avg}")

        # Initialize analyzer
        self.analyzer = ClusterAnalyzer(vocabulary, word_vectors, clusters, self.algorithm)

        return clusters, silhouette_avg

    def export_results(self, n_clusters, silhouette_score, dataset_name, clusters):
        """Export all clustering results."""
        vocabulary = self.loader.get_vocabulary()

        # Save clusters
        self.exporter.save_clusters_to_csv(vocabulary, clusters, f"clusters_{n_clusters}.csv")

        # Save cluster counts
        self.exporter.save_cluster_counts_to_csv(clusters, f"cluster_counts_{n_clusters}.csv")

        # Save silhouette scores
        timestamp = datetime.now().strftime("%d-%m-%Y-%H_%M")
        scaler = self.loader.get_scaler()
        self.exporter.save_silhouette_scores(
            silhouette_score, n_clusters, dataset_name,
            self.algorithm_name, self.distance_metric,
            scaler, f'silhouette_scores_{timestamp}.csv'
        )

        # Save centroids if available
        if hasattr(self.algorithm, 'get_cluster_centers'):
            self.exporter.save_centroids_to_csv(
                self.algorithm.get_cluster_centers(), f'centroids_{n_clusters}.csv'
            )

    def create_visualizations(self, n_clusters, clusters):
        """Create all visualizations."""
        vocabulary = self.loader.get_vocabulary()
        word_vectors = self.loader.get_word_vectors()

        # Basic cluster plot
        self.visualizer.plot_clusters(vocabulary, word_vectors, clusters,
                                      self.algorithm, f"clusters_{n_clusters}.html")

        # Centroids and paths plot
        self.visualizer.plot_centroids_and_paths(vocabulary, word_vectors, clusters,
                                                 self.algorithm, "centroids_paths.html")

        # Medoids plot with distances
        if self.analyzer:
            self.visualizer.plot_medoids_and_paths_with_distances(
                vocabulary, word_vectors, clusters, self.analyzer,
                "medoids_with_distances.html", self.exporter,
                "medoid_distances.csv", self.distance_metric
            )

    def run_automated_semantle_test(self, n_clusters, dataset_name, target_words=None,
                                    num_tests=5, max_guesses_per_word=100):
        """
        Run automated Semantle tests on the clustering.

        Args:
            n_clusters: number of clusters
            dataset_name: name of the dataset
            target_words: list of target words to test (if None, random selection)
            num_tests: number of tests to run if target_words is None
            max_guesses_per_word: maximum guesses allowed per word

        Returns:
            dict: aggregate results from all tests
        """
        # First, run the complete analysis
        clusters, silhouette_score = self.run_complete_analysis(n_clusters, dataset_name)

        vocabulary = self.loader.get_vocabulary()
        word_vectors = self.loader.get_word_vectors()

        # Select target words
        if target_words is None:
            # Randomly select words from vocabulary
            target_words = random.sample(vocabulary, min(num_tests, len(vocabulary)))
        else:
            # Filter to ensure words are in vocabulary
            target_words = [w for w in target_words if w in vocabulary]

        print(f"\n🎯 Running automated Semantle tests on {len(target_words)} words...")
        print("=" * 60)

        all_results = []

        for i, target_word in enumerate(target_words, 1):
            print(f"\n[Test {i}/{len(target_words)}] Target word: '{target_word}'")

            # Create output folder for this test
            test_folder = os.path.join(self.results_folder,
                                       f"semantle_test_{target_word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Create automated explorer
            explorer = AutomatedSemanticExplorer(
                analyzer=self.analyzer,
                word_vectors=word_vectors,
                vocabulary=vocabulary,
                full_vectors=word_vectors,
                full_vocab=vocabulary,
                full_clusters=clusters,
                base_output_folder=test_folder,
                target_word=target_word
            )

            # Run automated exploration
            result = explorer.run_automated_exploration(
                score_threshold=23,
                max_guesses=max_guesses_per_word
            )

            all_results.append(result)

            print(
                f"Completed test for '{target_word}': {'Found' if result['found'] else 'Not found'} in {result['total_guesses']} guesses")

        # Generate aggregate report
        aggregate_report = self.generate_aggregate_report(all_results, n_clusters)

        return aggregate_report

    def run_complete_analysis(self, n_clusters, dataset_name):
        """Run complete clustering analysis."""
        # Load data
        self.load_data()

        # Perform clustering
        clusters, silhouette_score = self.cluster(n_clusters)

        # Export results
        self.export_results(n_clusters, silhouette_score, dataset_name, clusters)

        # Create visualizations
        self.create_visualizations(n_clusters, clusters)

        return clusters, silhouette_score

    def generate_aggregate_report(self, all_results, n_clusters):
        """Generate an aggregate report from multiple test results."""
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results if r['found'])
        total_guesses = sum(r['total_guesses'] for r in all_results)

        report = {
            'n_clusters': n_clusters,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'average_guesses': total_guesses / total_tests if total_tests > 0 else 0,
            'individual_results': all_results
        }

        # Save aggregate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_folder, f"aggregate_report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write(f"Automated Semantle Testing - Aggregate Report\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Number of Clusters: {report['n_clusters']}\n")
            f.write(f"Total Tests: {report['total_tests']}\n")
            f.write(f"Successful Tests: {report['successful_tests']}\n")
            f.write(f"Success Rate: {report['success_rate']:.2%}\n")
            f.write(f"Average Guesses per Word: {report['average_guesses']:.2f}\n")
            f.write(f"\nIndividual Test Results:\n")
            f.write(f"{'-' * 60}\n")

            for i, result in enumerate(report['individual_results'], 1):
                f.write(f"\nTest {i}: {result['target_word']}\n")
                f.write(f"  Found: {result['found']}\n")
                f.write(f"  Total Guesses: {result['total_guesses']}\n")
                if result['best_guess']:
                    f.write(
                        f"  Best Guess: {result['best_guess']['word']} (Score: {result['best_guess']['score']:.2f})\n")

        # Also save as CSV for easier analysis
        results_df = pd.DataFrame([{
            'target_word': r['target_word'],
            'found': r['found'],
            'total_guesses': r['total_guesses'],
            'best_score': r['best_guess']['score'] if r['best_guess'] else 0,
            'best_word': r['best_guess']['word'] if r['best_guess'] else ''
        } for r in all_results])

        csv_path = os.path.join(self.results_folder, f"test_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"\n📊 Aggregate report saved to: {report_path}")
        print(f"📊 Results CSV saved to: {csv_path}")

        return report


# Example usage
if __name__ == "__main__":
    # Configuration
    word2vec_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\GoogleNews-vectors-negative300.bin"
    vocabulary_path = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\English-Words_Semantle.txt"
    results_folder = "C:\\Users\\yossi\\Downloads\\final project\\FinalProject\\MyData\\cluster_results_automated"

    # Parameters
    algorithm = 'kmeans'
    distance_metric = 'cosine'
    normalizer = 'standard'
    n_clusters = 10

    # Create automated system
    system = AutomatedVocabularyClusteringSystem(
        word2vec_path=word2vec_path,
        vocabulary_path=vocabulary_path,
        results_folder=results_folder,
        algorithm=algorithm,
        distance_metric=distance_metric,
        normalizer=normalizer
    )

    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(vocabulary_path))[0]

    # Option 1: Test with specific target words
    target_words = ['computer', 'dog', 'happy', 'science', 'music']

    # Option 2: Test with random words (set target_words=None)
    # target_words = None

    # Run automated tests
    results = system.run_automated_semantle_test(
        n_clusters=n_clusters,
        dataset_name=dataset_name,
        target_words=target_words,
        num_tests=5,  # Used only if target_words=None
        max_guesses_per_word=100
    )

    print(f"\n✅ Automated testing complete!")
    print(f"📊 Success rate: {results['success_rate']:.2%}")
    print(f"📊 Average guesses: {results['average_guesses']:.2f}")
    print(f"📂 Results saved to: {results_folder}")
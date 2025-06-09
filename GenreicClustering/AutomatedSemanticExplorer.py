import os
import pandas as pd
from datetime import datetime
from SemanticExplorer import SemanticExplorer
from SemantleSimulator import SemantleSimulator


class AutomatedSemanticExplorer(SemanticExplorer):
    """
    Extended SemanticExplorer that uses SemantleSimulator instead of user input.
    """

    def __init__(self, analyzer, word_vectors, vocabulary,
                 full_vectors=None, full_vocab=None, full_clusters=None,
                 base_output_folder="cluster_results", target_word=None):
        """
        Initialize the automated explorer with a Semantle simulator.

        Args:
            Same as SemanticExplorer, plus:
            target_word: the word to guess in the simulation
        """
        super().__init__(analyzer, word_vectors, vocabulary,
                         full_vectors, full_vocab, full_clusters,
                         base_output_folder)

        # Initialize the simulator
        self.simulator = SemantleSimulator(
            word_vectors=word_vectors,
            vocabulary=vocabulary,
            target_word=target_word
        )

        # Track performance metrics
        self.total_guesses = 0
        self.cluster_visits = {}

    def request_score_from_user(self, word):
        """
        Override the manual input method to use the simulator.

        Args:
            word: the word to guess

        Returns:
            float: similarity score from simulator
        """
        print(f"🤖 Auto-guessing word: '{word}'")

        # Get score from simulator
        result = self.simulator.make_guess(word)

        if result['score'] is None:
            print(f"❌ Word not in vocabulary: {word}")
            return 0.0

        score = result['score']
        self.total_guesses += 1

        print(f"📊 Score: {score:.2f}")

        # Check if we found the target
        if result['found']:
            print(f"🎉 Found target word '{word}' in {self.total_guesses} guesses!")

        return score

    def run_automated_exploration(self, score_threshold=23, max_guesses=100):
        """
        Run the exploration automatically using the simulator.

        Args:
            score_threshold: minimum score to trigger recursive clustering
            max_guesses: maximum number of guesses before stopping

        Returns:
            dict: results of the exploration
        """
        print(f"\n🚀 Starting automated exploration for target word: '{self.simulator.target_word}'")
        print(f"📋 Vocabulary size: {len(self.vocabulary)}")
        print(f"📊 Number of clusters: {len(set(self.analyzer.clusters))}")
        print("=" * 60)

        medoids = self.get_medoid_recommendations()
        guesses_made = 0
        best_guess = None
        best_score = -1

        while guesses_made < max_guesses:
            # Get next word suggestion
            word = self.suggest_next_medoid(medoids)

            if word is None:
                print("✅ No more medoid suggestions available.")
                break

            # Get score from simulator
            score = self.request_score_from_user(word)
            cluster = self.analyzer.find_word_cluster(word)

            # Track cluster visits
            if cluster not in self.cluster_visits:
                self.cluster_visits[cluster] = 0
            self.cluster_visits[cluster] += 1

            # Record guess
            self.guess_history.append({
                'word': word,
                'score': score,
                'cluster': cluster
            })

            guesses_made += 1

            # Track best guess
            if score > best_score:
                best_score = score
                best_guess = word

            print(f"✔️ Recorded: '{word}' | Cluster {cluster} | Score {score:.2f} | Guess #{guesses_made}")

            # Check if we found the target
            if score == 100:
                print(f"\n🎯 SUCCESS! Found target word '{word}' in {guesses_made} guesses!")
                break

            # Check if we should refine the cluster
            if score >= score_threshold:
                print(f"\n🎯 High score ({score:.2f})! Refining cluster {cluster}...")

                # Get sub-vocabulary for this cluster
                sub_vocab, sub_vectors = self.refine_cluster_data(
                    cluster, self.full_vocab, self.full_vectors, self.full_clusters
                )

                print(f"🔁 Cluster {cluster} contains {len(sub_vocab)} words.")

                # Check if target word is in this cluster
                if self.simulator.target_word in sub_vocab:
                    print(f"✅ Target word is in this cluster! Starting recursive exploration...")

                    # Create new simulator for sub-cluster
                    sub_simulator = SemantleSimulator(
                        word_vectors=sub_vectors,
                        vocabulary=sub_vocab,
                        target_word=self.simulator.target_word
                    )

                    # Run recursive clustering with automated exploration
                    self.recursive_clustering_automated(
                        sub_vocab, sub_vectors, depth=1,
                        target_simulator=sub_simulator
                    )
                else:
                    print(f"❌ Target word is NOT in this cluster. Continuing search...")

            print("-" * 40)

        # Generate final report
        results = self.generate_exploration_report()
        return results

    def recursive_clustering_automated(self, sub_vocab, sub_vectors, depth=1, target_simulator=None):
        """
        Automated version of recursive clustering.
        """
        print(f"\n🔬 Recursive Clustering Level {depth} (Automated)...")

        if len(sub_vocab) < 3:
            print("⛔ Not enough words to re-cluster.")
            return

        # Import required modules
        from KMeansClusterer import KMeansClusterer
        from ClusterAnalyzer import ClusterAnalyzer
        from Visualizer import Visualizer

        n_clusters = min(10, len(sub_vocab))
        kmeans = KMeansClusterer(distance_metric='euclidean')
        clusters = kmeans.fit(sub_vectors, n_clusters)
        sil_score = kmeans.calculate_silhouette_score(sub_vectors)

        print(f"📊 Sub-clustering complete (k={n_clusters}) | Silhouette Score: {sil_score:.4f}")

        new_analyzer = ClusterAnalyzer(sub_vocab, sub_vectors, clusters, kmeans)

        # Save results
        cluster_df = pd.DataFrame({"word": sub_vocab, "cluster": clusters})
        csv_path = os.path.join(self.output_folder, f"sub_clusters_level{depth}.csv")
        cluster_df.to_csv(csv_path, index=False)

        # Create visualizations
        visualizer = Visualizer(self.output_folder)
        visualizer.plot_clusters(sub_vocab, sub_vectors, clusters, kmeans,
                                 f"clusters_level{depth}.html")

        # Create automated sub-explorer
        sub_explorer = AutomatedSemanticExplorer(
            new_analyzer, sub_vectors, sub_vocab,
            full_vectors=sub_vectors, full_vocab=sub_vocab, full_clusters=clusters,
            base_output_folder=self.output_folder,
            target_word=self.simulator.target_word
        )

        # Use the provided simulator if available
        if target_simulator:
            sub_explorer.simulator = target_simulator

        # Run automated exploration on sub-cluster
        sub_explorer.run_automated_exploration()

    def generate_exploration_report(self):
        """
        Generate a comprehensive report of the exploration.
        """
        report = {
            'target_word': self.simulator.target_word,
            'total_guesses': self.total_guesses,
            'found': any(g['score'] == 100 for g in self.guess_history),
            'best_guess': max(self.guess_history, key=lambda x: x['score']) if self.guess_history else None,
            'cluster_visits': self.cluster_visits,
            'guess_history': self.guess_history
        }

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_folder, f"exploration_report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write(f"Automated Semantle Exploration Report\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Target Word: {report['target_word']}\n")
            f.write(f"Total Guesses: {report['total_guesses']}\n")
            f.write(f"Found: {report['found']}\n")

            if report['best_guess']:
                f.write(f"Best Guess: {report['best_guess']['word']} (Score: {report['best_guess']['score']:.2f})\n")

            f.write(f"\nCluster Visit Counts:\n")
            for cluster, count in sorted(report['cluster_visits'].items()):
                f.write(f"  Cluster {cluster}: {count} visits\n")

            f.write(f"\nGuess History:\n")
            for i, guess in enumerate(report['guess_history'], 1):
                f.write(f"  {i}. {guess['word']} - Score: {guess['score']:.2f}, Cluster: {guess['cluster']}\n")

        print(f"\n📄 Report saved to: {report_path}")

        return report
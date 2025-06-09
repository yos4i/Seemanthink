import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import random


class SemantleSimulator:
    """Simulates Semantle game mechanics for automated testing."""

    def __init__(self, word_vectors, vocabulary, target_word=None):
        """
        Initialize the Semantle simulator.

        Args:
            word_vectors: numpy array of word vectors
            vocabulary: list of words corresponding to vectors
            target_word: the word to guess (if None, randomly selected)
        """
        self.word_vectors = word_vectors
        self.vocabulary = vocabulary
        self.word_to_index = {word: i for i, word in enumerate(vocabulary)}

        # Set or randomly select target word
        if target_word and target_word in self.vocabulary:
            self.target_word = target_word
        else:
            self.target_word = random.choice(self.vocabulary)

        self.target_index = self.word_to_index[self.target_word]
        self.target_vector = self.word_vectors[self.target_index]

        # History of guesses
        self.guess_history = []

        print(f"Semantle Simulator initialized with target word: {self.target_word}")

    def calculate_similarity_score(self, guess_word):
        """
        Calculate similarity score between guess and target word.
        Uses the same formula as the original Semantle solver.

        Args:
            guess_word: the guessed word

        Returns:
            float: similarity score (0 to 100)
        """
        if guess_word not in self.vocabulary:
            return None

        guess_index = self.word_to_index[guess_word]
        guess_vector = self.word_vectors[guess_index]

        # Using the same calculation as in GloVe&W2V_Solver
        # (1 - cosine_distance) * 100
        from scipy.spatial.distance import cosine

        similarity = (1 - cosine(guess_vector, self.target_vector)) * 100

        return round(similarity, 2)

    def make_guess(self, word):
        """
        Make a guess and get the score.

        Args:
            word: the guessed word

        Returns:
            dict: guess result with word, score, and additional info
        """
        if word not in self.vocabulary:
            return {
                'word': word,
                'score': None,
                'error': 'Word not in vocabulary',
                'found': False
            }

        score = self.calculate_similarity_score(word)

        result = {
            'word': word,
            'score': score,
            'guess_number': len(self.guess_history) + 1,
            'found': word == self.target_word
        }

        self.guess_history.append(result)

        return result

    def get_hint(self, threshold=70):
        """
        Get a hint - returns a word with similarity above threshold.

        Args:
            threshold: minimum similarity score for hint

        Returns:
            str: a hint word or None if no suitable word found
        """
        # Calculate similarities for all words using the same formula
        hint_candidates = []
        guessed_words = {g['word'] for g in self.guess_history}

        for i, word in enumerate(self.vocabulary):
            if word in guessed_words or word == self.target_word:
                continue

            vector = self.word_vectors[i]
            similarity = (1 - cosine(vector, self.target_vector)) * 100

            if similarity >= threshold:
                hint_candidates.append((word, similarity))

        if hint_candidates:
            # Return the word with highest score
            hint_candidates.sort(key=lambda x: x[1], reverse=True)
            return hint_candidates[0][0]

        return None

    # def get_statistics(self):
    #     """Get game statistics."""
    #     if not self.guess_history:
    #         return None
    #
    #     stats = {
    #         'target_word': self.target_word,
    #         'total_guesses': len(self.guess_history),
    #         'found': any(g['found'] for g in
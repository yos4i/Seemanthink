import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

from sklearn.preprocessing import StandardScaler, Normalizer


class VocabularyLoader:
    """Handles loading and preprocessing of vocabulary data."""

    def __init__(self, word2vec_path, vocabulary_path, normalizer=None):
        self.word2vec_path = word2vec_path
        self.vocabulary_path = vocabulary_path
        self.wv = None
        self.vocabulary = []
        self.word_vectors = None
        if normalizer == 'standard':
            self.scaler = StandardScaler()
        elif normalizer == 'normalize':
            self.scaler = Normalizer()
        else:
            self.scaler = None

    def load_word2vec_model(self):
        """Load the Word2Vec model."""
        try:
            self.wv = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
            print(f"Word2Vec model loaded from {self.word2vec_path}")
        except Exception as e:
            raise Exception(f"Failed to load Word2Vec model: {e}")

    def load_vocabulary(self):
        """Load vocabulary from file and filter words present in Word2Vec model."""
        try:
            if self.wv is None:
                self.load_word2vec_model()

            with open(self.vocabulary_path, 'r') as file:
                words = file.read().splitlines()

            self.vocabulary = [word for word in words if word in self.wv]
            print(f"Loaded {len(self.vocabulary)} words from the vocabulary file.")

            # Extract word vectors
            self.word_vectors = np.array([self.wv[word] for word in self.vocabulary])

            # Apply normalization if specified
            if self.scaler:
                self.word_vectors = self.scaler.fit_transform(self.word_vectors)

        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.vocabulary_path} was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the vocabulary: {e}")

    def _apply_normalization(self):
        """Apply normalization to word vectors."""
        if self.normalizer == 'standard':
            self.scaler = StandardScaler()
            self.word_vectors = self.scaler.fit_transform(self.word_vectors)
        elif self.normalizer == 'normalize':
            self.scaler = Normalizer()
            self.word_vectors = self.scaler.fit_transform(self.word_vectors)
        else:
            self.scaler = None

    def get_vocabulary(self):
        """Return the loaded vocabulary."""
        return self.vocabulary

    def get_word_vectors(self):
        """Return the word vectors."""
        return self.word_vectors

    def get_scaler(self):
        """Return the scaler used for normalization."""
        return self.scaler













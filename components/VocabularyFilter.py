import re
import string
from typing import List, Set
from collections import Counter


class VocabularyFilter:
    
    def __init__(self):
        self.stopwords = self._get_stopwords()
        self.semantic_patterns = self._get_semantic_patterns()
        
    def _get_stopwords(self) -> Set[str]:
        """Extended stopwords list including common non-semantic words for Semantle."""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'do', 'does', 'did',
            'have', 'had', 'has', 'am', 'is', 'are', 'was', 'were', 'been',
            'being', 'this', 'these', 'that', 'those', 'i', 'you', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'our', 'their', 'myself', 'yourself', 'himself', 'herself',
            'ourselves', 'yourselves', 'themselves', 'who', 'whom', 'whose',
            'which', 'what', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now', 'up', 'out',
            'if', 'about', 'after', 'again', 'before', 'here', 'there',
            'over', 'under', 'down', 'off', 'above', 'below'
        }
        
        common_non_semantic = {
            'aa', 'aah', 'aahs', 'aal', 'ab', 'aba', 'abaca', 'abacas',
            'eh', 'em', 'en', 'er', 'es', 'ex', 'fa', 'go', 'ha', 'he',
            'hi', 'hm', 'ho', 'if', 'in', 'is', 'it', 'lo', 'ma', 'me',
            'mi', 'mm', 'mu', 'my', 'no', 'nu', 'od', 'oe', 'of', 'oh',
            'om', 'on', 'op', 'or', 'os', 'ow', 'ox', 'oy', 'pa', 'pe',
            'pi', 're', 'sh', 'si', 'so', 'ta', 'ti', 'to', 'uh', 'um',
            'un', 'up', 'us', 'ut', 'we', 'wo', 'xi', 'xu', 'ya', 'ye',
            'yo', 'za', 'al', 'am', 'an', 'ar', 'as', 'at', 'aw', 'ax',
            'ay', 'ba', 'be', 'bi', 'bo', 'by', 'da', 'de', 'do', 'ed',
            'ef', 'el', 'et', 'fe', 'gi', 'gu', 'id', 'jo', 'ka', 'ki',
            'la', 'li', 'na', 'ne', 'oi', 'ou', 'pu', 'qi', 'te', 'th',
            'ti', 'ug', 'ur', 'ut', 'wa', 'wu', 'ye', 'za', 'zo'
        }
        
        return stopwords.union(common_non_semantic)
    
    def _get_semantic_patterns(self) -> List[str]:
        """Patterns for non-semantic words to exclude."""
        return [
            r'^[aeiou]{2,}$',          # vowel clusters like 'aa', 'ooo'
            r'^[bcdfghjklmnpqrstvwxyz]{2,}$',  # consonant clusters
            r'^[a-z]{1,2}$',           # very short words (1-2 chars)
            r'^\d+$',                  # numbers only
            r'^[^a-zA-Z]',             # non-alphabetic start
            r'[^a-zA-Z]$',             # non-alphabetic end
            r'.*\d.*',                 # contains digits
        ]
    
    def filter_stopwords(self, words: List[str]) -> List[str]:
        """Remove stopwords from vocabulary."""
        return [word for word in words if word.lower() not in self.stopwords]
    
    def filter_by_length(self, words: List[str], min_length: int = 3, max_length: int = 20) -> List[str]:
        """Filter words by length."""
        return [word for word in words if min_length <= len(word) <= max_length]
    
    def filter_by_patterns(self, words: List[str]) -> List[str]:
        """Filter out words matching non-semantic patterns."""
        filtered = []
        for word in words:
            is_valid = True
            for pattern in self.semantic_patterns:
                if re.match(pattern, word.lower()):
                    is_valid = False
                    break
            if is_valid:
                filtered.append(word)
        return filtered
    
    def filter_by_frequency(self, words: List[str], min_freq: int = 2, max_freq: int = 10000) -> List[str]:
        """Filter words by frequency (requires word counts)."""
        word_counts = Counter(words)
        return [word for word in words if min_freq <= word_counts[word] <= max_freq]
    
    def filter_non_alphabetic(self, words: List[str]) -> List[str]:
        """Keep only words with alphabetic characters."""
        return [word for word in words if word.isalpha()]
    
    def filter_proper_nouns(self, words: List[str]) -> List[str]:
        """Remove obvious proper nouns (capitalized words)."""
        return [word for word in words if not word[0].isupper()]
    
    def apply_all_filters(self, words: List[str], 
                         min_length: int = 3, 
                         max_length: int = 20,
                         min_freq: int = 1,
                         max_freq: int = 50000,
                         remove_proper_nouns: bool = True) -> tuple:
        """Apply all filtering steps and return filtered words + statistics."""
        
        original_count = len(words)
        stats = {"original": original_count}
        
        # Step 1: Remove non-alphabetic
        words = self.filter_non_alphabetic(words)
        stats["after_alphabetic"] = len(words)
        
        # Step 2: Length filtering
        words = self.filter_by_length(words, min_length, max_length)
        stats["after_length"] = len(words)
        
        # Step 3: Pattern filtering
        words = self.filter_by_patterns(words)
        stats["after_patterns"] = len(words)
        
        # Step 4: Stopword removal
        words = self.filter_stopwords(words)
        stats["after_stopwords"] = len(words)
        
        # Step 5: Proper noun filtering (optional)
        if remove_proper_nouns:
            words = self.filter_proper_nouns(words)
            stats["after_proper_nouns"] = len(words)
        
        # Step 6: Remove duplicates
        words = list(set(words))
        stats["after_dedup"] = len(words)
        
        stats["final_count"] = len(words)
        stats["reduction_percentage"] = ((original_count - len(words)) / original_count) * 100
        
        return words, stats
    
    def print_filtering_stats(self, stats: dict):
        """Print detailed filtering statistics."""
        print("\n=== Vocabulary Filtering Statistics ===")
        print(f"Original vocabulary size: {stats['original']:,}")
        
        if 'after_alphabetic' in stats:
            removed = stats['original'] - stats['after_alphabetic']
            print(f"After removing non-alphabetic: {stats['after_alphabetic']:,} (-{removed:,})")
        
        if 'after_length' in stats:
            removed = stats['after_alphabetic'] - stats['after_length']
            print(f"After length filtering: {stats['after_length']:,} (-{removed:,})")
        
        if 'after_patterns' in stats:
            removed = stats['after_length'] - stats['after_patterns']
            print(f"After pattern filtering: {stats['after_patterns']:,} (-{removed:,})")
        
        if 'after_stopwords' in stats:
            removed = stats['after_patterns'] - stats['after_stopwords']
            print(f"After stopword removal: {stats['after_stopwords']:,} (-{removed:,})")
        
        if 'after_proper_nouns' in stats:
            removed = stats['after_stopwords'] - stats['after_proper_nouns']
            print(f"After proper noun removal: {stats['after_proper_nouns']:,} (-{removed:,})")
        
        if 'after_dedup' in stats:
            removed = stats.get('after_proper_nouns', stats['after_stopwords']) - stats['after_dedup']
            print(f"After deduplication: {stats['after_dedup']:,} (-{removed:,})")
        
        print(f"\nFinal vocabulary size: {stats['final_count']:,}")
        print(f"Total reduction: {stats['reduction_percentage']:.1f}%")
        print("=" * 40)


def demo_filter_vocabulary(vocabulary_file_path: str):
    """Demo function to filter a vocabulary file."""
    
    filter_system = VocabularyFilter()
    
    try:
        with open(vocabulary_file_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(words)} words from {vocabulary_file_path}")
        
        # Apply all filters
        filtered_words, stats = filter_system.apply_all_filters(
            words, 
            min_length=3, 
            max_length=20,
            remove_proper_nouns=True
        )
        
        # Print statistics
        filter_system.print_filtering_stats(stats)
        
        # Show sample of filtered words
        print("\nSample filtered words:")
        print(", ".join(filtered_words[:20]))
        
        # Optionally save filtered vocabulary
        output_path = vocabulary_file_path.replace('.txt', '_filtered.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in sorted(filtered_words):
                f.write(f"{word}\n")
        
        print(f"\nFiltered vocabulary saved to: {output_path}")
        
        return filtered_words, stats
        
    except FileNotFoundError:
        print(f"Error: Could not find vocabulary file: {vocabulary_file_path}")
        return None, None


if __name__ == "__main__":
    # Example usage - you'll need to provide the correct path to your vocabulary file
    vocabulary_path = "path/to/your/vocabulary.txt"
    demo_filter_vocabulary(vocabulary_path)
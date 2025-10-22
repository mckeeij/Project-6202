"""
vcode_nlp_processor.py
----------------
A comprehensive NLP text processor using NLTK, NumPy, and Matplotlib.

It performs:
- Text loading and cleaning
- Tokenization and lemmatization
- Stopword removal
- POS tagging and Named Entity Recognition
- Frequency distribution and word cloud plotting
- Summary statistics
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

# Ensure all NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

class NLPProcessor:
    def __init__(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.text = self._load_text()
        self.tokens = []
        self.filtered_tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.named_entities = []
        self.freq_dist = None

    def _load_text(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            return f.read()

    def clean_text(self):
        """Basic cleaning: remove non-alphabetic chars and lowercase."""
        self.text = re.sub(r"[^a-zA-Z\s]", "", self.text)
        self.text = self.text.lower()

    def tokenize(self):
        """Tokenize text into words."""
        self.tokens = word_tokenize(self.text)

    def remove_stopwords(self):
        """Remove common stopwords."""
        stop_words = set(stopwords.words('english'))
        self.filtered_tokens = [w for w in self.tokens if w not in stop_words and len(w) > 1]

    def lemmatize(self):
        """Lemmatize words using POS tagging for accuracy."""
        lemmatizer = WordNetLemmatizer()

        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        pos_tags = pos_tag(self.filtered_tokens)
        self.lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tags]
        self.pos_tags = pos_tags

    def analyze_named_entities(self):
        """Extract named entities using NLTK's NE chunker."""
        chunked = ne_chunk(self.pos_tags)
        self.named_entities = []
        for subtree in chunked:
            if hasattr(subtree, 'label'):
                entity = " ".join([token for token, pos in subtree.leaves()])
                self.named_entities.append((entity, subtree.label()))

    def compute_frequencies(self):
        """Compute frequency distribution of lemmatized words."""
        self.freq_dist = FreqDist(self.lemmas)
        return self.freq_dist

    def plot_top_words(self, n=20):
        """Plot top N most frequent words."""
        if not self.freq_dist:
            self.compute_frequencies()

        top_words = self.freq_dist.most_common(n)
        words, counts = zip(*top_words)
        y_pos = np.arange(len(words))

        plt.figure(figsize=(10, 6))
        plt.barh(y_pos, counts)
        plt.yticks(y_pos, words)
        plt.xlabel("Frequency")
        plt.title(f"Top {n} Most Frequent Words")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print a summary of text statistics."""
        total_words = len(self.tokens)
        unique_words = len(set(self.tokens))
        lexical_richness = unique_words / total_words if total_words > 0 else 0

        print("📊 TEXT SUMMARY 📊")
        print(f"File: {os.path.basename(self.filepath)}")
        print(f"Total words: {total_words}")
        print(f"Unique words: {unique_words}")
        print(f"Lexical richness: {lexical_richness:.3f}")
        print(f"Named Entities Found: {len(self.named_entities)}")
        print("Sample entities:", self.named_entities[:10])
        print("\nMost common words:")
        print(self.freq_dist.most_common(10))

    def run_full_analysis(self):
        """Execute full NLP pipeline."""
        print("Running full NLP pipeline...\n")
        self.clean_text()
        self.tokenize()
        self.remove_stopwords()
        self.lemmatize()
        self.analyze_named_entities()
        self.compute_frequencies()
        self.summary()
        self.plot_top_words()

if __name__ == "__main__":
    # Example usage
    filepath = input("Enter path to .txt file: ").strip()
    nlp = NLPProcessor(filepath)
    nlp.run_full_analysis()

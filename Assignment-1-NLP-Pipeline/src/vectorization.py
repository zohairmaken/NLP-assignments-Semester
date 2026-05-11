"""
vectorization.py
----------------
This module handles feature engineering—converting text into numbers.
Machine Learning models can't understand text directly; they need numeric vectors.

Methods:
1. Bag of Words (CountVectorizer): Counts how many times each word appears.
2. TF-IDF (TfidfVectorizer): Weights words by importance (Term Frequency - Inverse Document Frequency).
    - High weight: Words unique to a specific review.
    - Low weight: Words common across all reviews.

Why this matters in viva:
  - Mention that BoW focus on frequency, while TF-IDF helps filter out common words that
    don't provide distinct information.
  - TF-IDF is generally better for shorter, informal texts like reviews.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from typing import Tuple

class TextVectorizer:
    def __init__(self, method='tfidf', max_features=1000):
        self.method = method
        self.max_features = max_features
        
        if method == 'bow':
            self.vectorizer = CountVectorizer(max_features=max_features)
        else:
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            
    def fit_transform(self, texts: list) -> Tuple:
        """Fit vectorizer to text and return the sparse matrix."""
        X = self.vectorizer.fit_transform(texts)
        return X, self.vectorizer
    
    def transform(self, texts: list):
        """Transform new text using the fitted vectorizer."""
        return self.vectorizer.transform(texts)

def compare_vectors(texts: list):
    """
    Demonstration function to compare BoW and TF-IDF for the same text.
    Useful for explaining to the external examiner.
    """
    # 1. Bag of Words
    bow_vec = CountVectorizer()
    bow_res = bow_vec.fit_transform(texts)
    bow_df = pd.DataFrame(bow_res.toarray(), columns=bow_vec.get_feature_names_out())
    
    # 2. TF-IDF
    tfidf_vec = TfidfVectorizer()
    tfidf_res = tfidf_vec.fit_transform(texts)
    tfidf_df = pd.DataFrame(tfidf_res.toarray(), columns=tfidf_vec.get_feature_names_out())
    
    return bow_df, tfidf_df

if __name__ == "__main__":
    test_texts = [
        "the product is great",
        "the delivery is great"
    ]
    bow, tfidf = compare_vectors(test_texts)
    
    print("BAG OF WORDS (COUNTS)")
    print(bow)
    print("\nTF-IDF (WEIGHTED)")
    print(tfidf)
    print("\nNotice how 'the', 'is', 'great' get lower weights in TF-IDF because they are common.")

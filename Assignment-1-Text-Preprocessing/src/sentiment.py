"""
sentiment.py
------------
Handles sentiment prediction using two different techniques:
1. Rule-Based (Lexicon): Uses pre-defined lists of good/bad words.
2. Machine Learning: Multinomial Naive Bayes trained on logic.

Naive Bayes is chosen for the assignment because:
- It's highly explainable.
- It works extremely well with text data (word frequencies).
- It's much faster than deep learning.

Why this matters in viva:
  - Explaining Naive Bayes: "It calculates the probability of a sentiment 
    given the words in the review using Bayes' Theorem."
"""

from sklearn.naive_bayes import MultinomialNB
from .preprocessing import preprocess_pipeline
from .vectorization import TextVectorizer
import pandas as pd

class RuleBasedSentiment:
    """
    A simple lexicographic approach. 
    Good for explaining basic logic in a viva.
    """
    def __init__(self):
        self.positive_words = {'good', 'great', 'amazing', 'excellent', 'happy', 'love', 'satisfied', 'acha', 'pasand', 'theek'}
        self.negative_words = {'bad', 'worst', 'poor', 'disappointed', 'terrible', 'waste', 'broken', 'late', 'bekaar', 'kharab'}

    def predict(self, text: str) -> str:
        text = text.lower()
        pos_count = sum(1 for word in text.split() if word in self.positive_words)
        neg_count = sum(1 for word in text.split() if word in self.negative_words)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

class MLSentimentModel:
    """
    Machine Learning based sentiment classifier.
    """
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = TextVectorizer(method='tfidf')
        
    def train(self, df: pd.DataFrame):
        """Train NB model using TF-IDF features."""
        # Preprocess all reviews
        processed_reviews = df['review'].apply(preprocess_pipeline)
        
        # Vectorize
        X, _ = self.vectorizer.fit_transform(processed_reviews)
        y = df['sentiment']
        
        # Train
        self.model.fit(X, y)
        print("[sentiment] ML Model trained successfully.")

    def predict(self, text: str) -> str:
        """Predict sentiment for new text."""
        clean = preprocess_pipeline(text)
        X = self.vectorizer.transform([clean])
        return self.model.predict(X)[0]

# --- Testing ---
if __name__ == "__main__":
    from .utils import load_sample_dataset
    
    df = load_sample_dataset()
    
    # 1. Test Rule-Based
    rb = RuleBasedSentiment()
    print(f"Rule Based ('worst product'): {rb.predict('worst product')}")
    
    # 2. Test ML-Based
    ml = MLSentimentModel()
    ml.train(df)
    print(f"ML Based ('best item'): {ml.predict('best item ever')}")

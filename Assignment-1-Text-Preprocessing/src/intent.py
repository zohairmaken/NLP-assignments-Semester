"""
intent.py
---------
Classifies the 'Reason' or 'Intent' behind a customer review.
Common intents in E-commerce:
- Complaint (Damaged product, wrong item)
- Refund Request (Money back)
- Delivery Issue (Late shipping, tracking problems)
- General Query (Asking about price, availability)

Technique: Logistic Regression.
Logistic Regression is great for multi-class classification and is
very easy to explain: "It calculates the probability of each class
and picks the one with the highest score."

Why this matters in viva:
  - Explain that 'Intent' helps companies automatically route tickets
    to the right department (e.g., Refund requests to the Finance team).
"""

from sklearn.linear_model import LogisticRegression
from .preprocessing import preprocess_pipeline
from .vectorization import TextVectorizer
import pandas as pd

class IntentClassifier:
    def __init__(self):
        # Logistic Regression is stable for multi-class problems
        self.model = LogisticRegression(max_iter=500)
        self.vectorizer = TextVectorizer(method='tfidf')

    def train(self, df: pd.DataFrame):
        """Train Intent Classifier."""
        # Preprocess
        processed_reviews = df['review'].apply(preprocess_pipeline)
        
        # Vectorize
        X, _ = self.vectorizer.fit_transform(processed_reviews)
        y = df['intent']
        
        # Train
        self.model.fit(X, y)
        print("[intent] Intent Model trained successfully.")

    def predict(self, text: str) -> str:
        """Predict intent for new text."""
        clean = preprocess_pipeline(text)
        X = self.vectorizer.transform([clean])
        return self.model.predict(X)[0]

if __name__ == "__main__":
    from .utils import load_sample_dataset
    df = load_sample_dataset()
    
    ic = IntentClassifier()
    ic.train(df)
    print(f"Prediction ('where is my parcel'): {ic.predict('where is my parcel')}")
    print(f"Prediction ('I want my money back'): {ic.predict('I want my money back')}")

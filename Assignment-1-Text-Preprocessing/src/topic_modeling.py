"""
topic_modeling.py
-----------------
Finds hidden "topics" or "themes" in a collection of reviews without labels.
Technique: Non-Negative Matrix Factorization (NMF).

Logic:
1. Matrix Factorization splits the big Document-Word matrix into two:
    - Document-Topic matrix (which topic does this review belong to?)
    - Topic-Word matrix (which words define this topic?)
2. NMF is preferred over LDA for short texts because it is faster and 
   sometimes produces more readable results.

Why this matters in viva:
  - Topic modeling is "Unsupervised Learning" because we don't need labels.
  - It helps business owners see what customers are talking about (e.g., Size, Battery, Shipping).
"""

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import preprocess_pipeline

class TopicModeler:
    def __init__(self, n_topics=3):
        self.n_topics = n_topics
        self.nmf = NMF(n_components=n_topics, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.feature_names = None

    def fit(self, texts: list):
        """Build the topic model from a list of reviews."""
        # 1. Preprocess
        clean_texts = [preprocess_pipeline(t) for t in texts]
        
        # 2. Vectorize (standard TF-IDF)
        tfidf_sparse = self.vectorizer.fit_transform(clean_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # 3. Fit NMF
        self.nmf.fit(tfidf_sparse)
        print(f"[topics] NMF model fitted with {self.n_topics} topics.")

    def get_topics(self, n_top_words=10):
        """Extract keywords for each topic."""
        topics = {}
        for topic_idx, topic in enumerate(self.nmf.components_):
            # Get highest weighing indices
            top_indices = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics[f"Topic {topic_idx + 1}"] = top_words
        return topics

    def predict_topic(self, text: str):
        """Identify which topic a specific review belongs to."""
        clean = preprocess_pipeline(text)
        vec = self.vectorizer.transform([clean])
        # Returns index of the highest probability topic
        topic_weights = self.nmf.transform(vec)
        return topic_weights.argmax() + 1

if __name__ == "__main__":
    from .utils import load_sample_dataset
    df = load_sample_dataset()
    
    tm = TopicModeler(n_topics=3)
    tm.fit(df['review'].tolist())
    
    for t, words in tm.get_topics(5).items():
        print(f"{t}: {', '.join(words)}")

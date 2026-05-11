"""
main.py
-------
The orchestrator of the entire NLP pipeline. 
Running this script will:
1. Load and prepare the data.
2. Train the Sentiment and Intent models.
3. Perform Topic Modeling.
4. Evaluate performance and save results.
5. Save the trained models for the interface to use.

Why this matters in viva:
  - This is the "Main Entry Point". It shows how separate modules 
    talk to each other to solve a complete problem.
"""

import os
from src.utils import load_sample_dataset, save_model, print_section, REPORTS_DIR
from src.sentiment import MLSentimentModel
from src.intent import IntentClassifier
from src.topic_modeling import TopicModeler
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def run_pipeline():
    print_section("Customer Reviews Intelligence System")
    
    # 1. Data Loading
    df = load_sample_dataset()
    print(f"Dataset Loaded. Total reviews: {len(df)}")
    
    # Split for evaluation
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 2. Sentiment Analysis Pipeline
    print_section("Training Sentiment Model")
    sent_model = MLSentimentModel()
    sent_model.train(train_df)
    
    y_true_sent = test_df['sentiment']
    y_pred_sent = test_df['review'].apply(sent_model.predict)
    
    evaluate_model(
        y_true_sent, 
        y_pred_sent, 
        model_name="Sentiment NB", 
        save_path=os.path.join(REPORTS_DIR, "figures", "sentiment_cm.png")
    )
    
    # 3. Intent Classification Pipeline
    print_section("Training Intent Classifier")
    intent_model = IntentClassifier()
    intent_model.train(train_df)
    
    y_true_int = test_df['intent']
    y_pred_int = test_df['review'].apply(intent_model.predict)
    
    evaluate_model(
        y_true_int, 
        y_pred_int, 
        model_name="Intent LogReg", 
        save_path=os.path.join(REPORTS_DIR, "figures", "intent_cm.png")
    )
    
    # 4. Topic Modeling
    print_section("Topic Modeling (NMF)")
    topic_modeler = TopicModeler(n_topics=3)
    topic_modeler.fit(df['review'].tolist())
    
    print("\nIdentified Topics and Keywords:")
    for topic, words in topic_modeler.get_topics(8).items():
        print(f"👉 {topic}: {', '.join(words)}")
        
    # 5. Saving Models
    print_section("Saving Models for Production")
    save_model(sent_model, "sentiment_model.pkl")
    save_model(intent_model, "intent_model.pkl")
    save_model(topic_modeler, "topic_modeler.pkl")
    
    print("\n[main] Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()

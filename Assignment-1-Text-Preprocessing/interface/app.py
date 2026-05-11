"""
app.py (Gradio Interface)
-------------------------
This script creates a web interface for the project.
It allows anyone to test the models without looking at the code.

Why Gradio?
- It's specifically designed for ML demos.
- It's "Low Code" and highly professional.

Why this matters in viva:
  - Explain that real-world ML models must be accessible to non-technical users.
  - An interface makes the code feel like a "System" rather than just a script.
"""

import gradio as gr
import os
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model, print_section

# Global variables to hold models
SENT_MODEL = None
INTENT_MODEL = None
TOPIC_MODEL = None

def load_all_models():
    global SENT_MODEL, INTENT_MODEL, TOPIC_MODEL
    try:
        SENT_MODEL = load_model("sentiment_model.pkl")
        INTENT_MODEL = load_model("intent_model.pkl")
        TOPIC_MODEL = load_model("topic_modeler.pkl")
        return True
    except Exception as e:
        print(f"[app] Error loading models: {e}")
        return False

def analyze_review(review_text):
    if not review_text.strip():
        return "Please enter a review...", "", ""
    
    if SENT_MODEL is None:
        return "Models not trained. Please run main.py first.", "", ""
    
    # 1. Predict Sentiment
    sentiment = SENT_MODEL.predict(review_text)
    
    # 2. Predict Intent
    intent = INTENT_MODEL.predict(review_text)
    
    # 3. Analyze Topic
    topic_idx = TOPIC_MODEL.predict_topic(review_text)
    # Get keywords for that topic
    topics_dict = TOPIC_MODEL.get_topics(5)
    keywords = ", ".join(topics_dict[f"Topic {topic_idx}"])
    topic_info = f"Topic {topic_idx} (Keywords: {keywords})"
    
    # Formatting for UI
    sentiment_emoji = "😊 Positive" if sentiment == "positive" else "☹️ Negative" if sentiment == "negative" else "😐 Neutral"
    intent_formatted = intent.replace("_", " ").title()
    
    return sentiment_emoji, intent_formatted, topic_info

# Build Interface
def build_interface():
    # Load models on startup
    models_ready = load_all_models()
    
    with gr.Blocks(title="Customer Reviews Intelligence") as demo:
        gr.Markdown("# 🤖 Customer Reviews Intelligence System")
        gr.Markdown("Submit a customer review to analyze its **Sentiment**, **Intent**, and **Topic** using NLP.")
        
        if not models_ready:
            gr.Warning("⚠️ Models not found! Please run `python main.py` first to train and save models.")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Enter Customer Review", 
                    placeholder="e.g., The product is amazing but delivery was late...",
                    lines=4
                )
                submit_btn = gr.Button("Analyze Intelligence", variant="primary")
                
                gr.Examples(
                    examples=[
                        ["The quality is great and I love the product!"],
                        ["Where is my order? It's been two weeks."],
                        ["I want a full refund, item arrived broken."],
                        ["Mera product bht acha hai, delivery time pe thi."], # Roman Urdu example
                        ["Bekar product, money waste ho gaya."] # Roman Urdu example
                    ],
                    inputs=input_text
                )
            
            with gr.Column():
                out_sent = gr.Label(label="Predicted Sentiment")
                out_intent = gr.Textbox(label="Identified Intent", interactive=False)
                out_topic = gr.Textbox(label="Topic Discovery", interactive=False)
        
        submit_btn.click(
            fn=analyze_review, 
            inputs=input_text, 
            outputs=[out_sent, out_intent, out_topic]
        )
        
        gr.Markdown("---")
        gr.Markdown("### 🎓 Academic Project Notes")
        gr.Markdown("""
        - **Models:** Multinomial Naive Bayes (Sentiment) & Logistic Regression (Intent)
        - **Preprocessing:** NLTK based pipeline (Cleaning, Stemming, Stopwords)
        - **Unsupervised:** NMF used for Topic Discovery
        """)

    return demo

if __name__ == "__main__":
    app = build_interface()
    # Launch with share=False for local, True for public link (requires internet)
    app.launch()

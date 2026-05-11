"""
utils.py
--------
Utility functions for the Customer Reviews NLP project.
This module handles:
  - Generating a synthetic/demo dataset (since real datasets require API keys)
  - Providing helper functions used across multiple modules
  - Saving and loading trained models

Why this matters in viva:
  The utils module is the "backbone" of every professional ML project.
  It avoids code duplication and keeps each module focused on one task.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# ──────────────────────────────────────────────
# 1. PATHS – centralise all file paths here
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
DATA_AUG    = os.path.join(BASE_DIR, "data", "augmented")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure all directories exist when this file is imported
for _dir in [DATA_RAW, DATA_PROC, DATA_AUG, MODELS_DIR, REPORTS_DIR,
             os.path.join(REPORTS_DIR, "figures")]:
    os.makedirs(_dir, exist_ok=True)


# ──────────────────────────────────────────────
# 2. SYNTHETIC DATASET
#    We create a labelled dataset manually so the project can run
#    without downloading from Kaggle/HuggingFace.
#    Each row has: review text, sentiment label, intent label.
# ──────────────────────────────────────────────

REVIEWS_DATA = [
    # ---------- POSITIVE SENTIMENT ----------
    ("The product quality is amazing, totally worth it!", "positive", "general_query"),
    ("Fast delivery and great packaging, very satisfied!", "positive", "general_query"),
    ("Excellent customer service, resolved my issue quickly.", "positive", "general_query"),
    ("Love this item! Will definitely buy again.", "positive", "general_query"),
    ("Best purchase I've made this year. Highly recommend.", "positive", "general_query"),
    ("Product arrived on time and works perfectly.", "positive", "general_query"),
    ("Really happy with the quality. Five stars!", "positive", "general_query"),
    ("Super fast shipping and the item is exactly as described.", "positive", "general_query"),
    ("Great value for money, I am impressed.", "positive", "general_query"),
    ("Absolutely love it! Packaging was also very nice.", "positive", "general_query"),
    ("Bohot acha product hai, bilkul sahi aya!", "positive", "general_query"),       # Roman Urdu
    ("Mujhe bohat pasand aaya yeh product.", "positive", "general_query"),             # Roman Urdu
    ("Shukriya! Delivery bhi time pe thi.", "positive", "general_query"),              # Roman Urdu
    ("Product theek tha, delivery bhi fast thi.", "positive", "general_query"),       # Roman Urdu
    ("10/10 love it!!!", "positive", "general_query"),
    ("omg this is sooo good 😍", "positive", "general_query"),
    ("woww amazingg item received fast", "positive", "general_query"),

    # ---------- NEGATIVE SENTIMENT ----------
    ("Worst product ever, broke after two days.", "negative", "complaint"),
    ("Very disappointed, the color was completely different.", "negative", "complaint"),
    ("Do not buy this! Total waste of money.", "negative", "complaint"),
    ("Poor quality, the stitching came off immediately.", "negative", "complaint"),
    ("Packaging was damaged and the item was scratched.", "negative", "complaint"),
    ("Customer support is terrible, no one picks up.", "negative", "complaint"),
    ("Item never arrived, very frustrating.", "negative", "delivery_issue"),
    ("Delivery is extremely late, no updates from seller.", "negative", "delivery_issue"),
    ("Three weeks and still waiting for my order!", "negative", "delivery_issue"),
    ("Wrong product sent, this is unacceptable!", "negative", "complaint"),
    ("Bekaar product hai, bilkul se kaam nahi kiya.", "negative", "complaint"),        # Roman Urdu
    ("Aik haftay baad bhi delivery nahi aayi.", "negative", "delivery_issue"),        # Roman Urdu
    ("Kharab quality thi, bilkul bhi pasand nahi aaya.", "negative", "complaint"),    # Roman Urdu
    ("total waste!!! dont buy!!!!", "negative", "complaint"),
    ("garbage product omg", "negative", "complaint"),
    ("arrived broken n nobody is helping me", "negative", "complaint"),

    # ---------- NEUTRAL SENTIMENT ----------
    ("The product is okay, nothing special.", "neutral", "general_query"),
    ("It works, but the quality could be better.", "neutral", "general_query"),
    ("Average product. Does the job.", "neutral", "general_query"),
    ("Delivery was on time. Product is decent.", "neutral", "general_query"),
    ("Not great not terrible. Just okay.", "neutral", "general_query"),
    ("Could be better but it does what it says.", "neutral", "general_query"),
    ("Normal quality. Expected more for the price.", "neutral", "general_query"),
    ("Theek theek hai, acha bhi nahi bura bhi nahi.", "neutral", "general_query"),   # Roman Urdu

    # ---------- REFUND REQUESTS ----------
    ("I want a full refund. This product is defective.", "negative", "refund_request"),
    ("Please process my refund immediately.", "negative", "refund_request"),
    ("I need to return this item and get my money back.", "negative", "refund_request"),
    ("Requesting a refund for order #12345.", "negative", "refund_request"),
    ("How do I return this product? Please help.", "negative", "refund_request"),
    ("Mujhe wapis paise chahiye, product kharab tha.", "negative", "refund_request"), # Roman Urdu
    ("Return karna chahta hoon, please guide karein.", "negative", "refund_request"), # Roman Urdu
    ("Refund nahi aaya abhi tak, kab milega?", "negative", "refund_request"),         # Roman Urdu

    # ---------- DELIVERY ISSUES ----------
    ("Where is my order? It's been 10 days!", "negative", "delivery_issue"),
    ("Delivery tracking shows delivered but I received nothing.", "negative", "delivery_issue"),
    ("The package was left at the wrong address.", "negative", "delivery_issue"),
    ("Delivery person was very rude and threw the package.", "negative", "delivery_issue"),
    ("Still waiting for delivery update from three weeks.", "negative", "delivery_issue"),
    ("Mera parcel kahan hai? 2 hafte ho gaye hain.", "negative", "delivery_issue"),   # Roman Urdu
    ("Delivery bahut late ho gayi, koi update nahi.", "negative", "delivery_issue"),  # Roman Urdu

    # ---------- GENERAL QUERIES ----------
    ("Can you tell me if this item is available in blue?", "neutral", "general_query"),
    ("What is the return policy for this product?", "neutral", "general_query"),
    ("Is this product compatible with iPhone 13?", "neutral", "general_query"),
    ("What are the dimensions of this product?", "neutral", "general_query"),
    ("How long does delivery take to Karachi?", "neutral", "general_query"),
    ("Is there a warranty with this product?", "neutral", "general_query"),
    ("Yeh product kahan se mila? Link share karein.", "neutral", "general_query"),    # Roman Urdu
    ("Delivery time kya hai Lahore ke liye?", "neutral", "general_query"),            # Roman Urdu
]


def load_sample_dataset() -> pd.DataFrame:
    """
    Load the built-in synthetic dataset.

    Returns
    -------
    pd.DataFrame
        Columns: 'review', 'sentiment', 'intent'

    Note: In a real project you would replace this with:
        pd.read_csv("data/raw/reviews.csv")
    """
    df = pd.DataFrame(REVIEWS_DATA, columns=["review", "sentiment", "intent"])
    df.index.name = "id"

    # Save a copy to data/raw for reference
    raw_path = os.path.join(DATA_RAW, "sample_reviews.csv")
    df.to_csv(raw_path, index=False)
    print(f"[utils] Dataset saved to {raw_path}")
    return df


def save_model(model, filename: str) -> str:
    """
    Persist a trained model to the models/ directory using pickle.

    Parameters
    ----------
    model : any sklearn estimator or vectorizer
    filename : str  e.g. 'sentiment_model.pkl'

    Returns
    -------
    str : full path where the model was saved
    """
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[utils] Model saved → {path}")
    return path


def load_model(filename: str):
    """
    Load a previously saved model from the models/ directory.

    Parameters
    ----------
    filename : str  e.g. 'sentiment_model.pkl'

    Returns
    -------
    Loaded model object
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[utils] Model loaded ← {path}")
    return model


def timestamp() -> str:
    """Return a human-readable timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_section(title: str) -> None:
    """Print a clearly formatted section header for console output."""
    print("\n" + "=" * 60)
    print(f"  {title.upper()}")
    print("=" * 60)

"""
preprocessing.py
----------------
This module implements the text cleaning and preparation pipeline.
In NLP, preprocessing is crucial because "Garbage In, Garbage Out".

Standard Steps:
1. Lowercasing: Ensures that 'Great' and 'great' are treated the same.
2. Cleaning: Removes URLs, emojis, and special characters.
3. Tokenization: Splitting text into individual words.
4. Stop Word Removal: Removing common words (the, is, in) that don't add much meaning.
5. Roman Urdu Handling: Basic normalization for non-standard spelling.

Why this matters in viva:
  - Mention that preprocessing reduces the "vocabulary size", making models faster and more accurate.
  - Explain that for sentiment, we might keep words like 'not' which are often listed as stop words.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Required Downloads for NLTK ---
# These will run the first time the project starts.
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"[preprocessing] NLTK download warning: {e}")

# Define punctuation and stopwords globally for efficiency
STOP_WORDS = set(stopwords.words('english'))
# Optional: Remove 'not' and 'no' from stop words because they change sentiment
SENTIMENT_WORDS = {'not', 'no', 'never', 'none', 'neither', 'nor', 'but', 'against'}
CLEAN_STOP_WORDS = STOP_WORDS - SENTIMENT_WORDS

def clean_text(text: str) -> str:
    """
    Apply basic cleaning: lowercasing, removing URLs, emojis, and special chars.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove punctuation and special characters (keeping some for Roman Urdu)
    # We substitute everything that isn't a word character or whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 4. Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def normalize_roman_urdu(text: str) -> str:
    """
    Very simple normalization for Roman Urdu common variations.
    Real Roman Urdu normalization is hard, but this shows 'intent' for the exam.
    """
    replacements = {
        r'\baay\b': 'aye',
        r'\bbht\b': 'bohat',
        r'\bboht\b': 'bohat',
        r'\bkch\b': 'kuch',
        r'\bny\b': 'nai',
        r'\bna\b': 'nai',
        r'\bh\b': 'hai',
        r'\bhein\b': 'hain',
        r'\bjst\b': 'just',
        r'\bmje\b': 'mujhe',
        r'\bmujhy\b': 'mujhe',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def preprocess_pipeline(text: str, remove_stops: bool = True) -> str:
    """
    Full pipeline for text preparation.
    """
    # 1. Basic Cleaning
    text = clean_text(text)
    
    # 2. Roman Urdu normalization
    text = normalize_roman_urdu(text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stop word removal
    if remove_stops:
        tokens = [word for word in tokens if word not in CLEAN_STOP_WORDS]
    
    return " ".join(tokens)

if __name__ == "__main__":
    # Test cases for the examiner
    samples = [
        "OMG!! This is amazing... check it out: https://xyz.com 😍",
        "Muje yeh product bht pasand aaya! 10/10",
        "The delivery was NOT good, very disappointed."
    ]
    
    print("DEMONSTRATION OF PREPROCESSING PIPELINE")
    print("-" * 40)
    for s in samples:
        processed = preprocess_pipeline(s)
        print(f"Original:  {s}")
        print(f"Processed: {processed}")
        print("-" * 40)

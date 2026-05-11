import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    return ' '.join(cleaned_words)

def load_and_preprocess(file_path):
    """
    Loads the dataset and applies cleaning.
    """
    try:
        df = pd.read_csv(file_path)
        df['cleaned_text'] = df['text'].apply(clean_text)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

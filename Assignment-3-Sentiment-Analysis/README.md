<div align="center">

<h1>💬 Assignment 3 — Sentiment Analysis</h1>

<p><em>Classical NLP-based sentiment classification with feature engineering and model evaluation</em></p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-4B8BBE?style=flat-square)](https://www.nltk.org/)

</div>

---

## 📌 Objective

Implement a complete sentiment analysis pipeline using classical NLP techniques to classify text documents (reviews / opinions) as **Positive**, **Negative**, or **Neutral**.

The assignment demonstrates the full ML workflow — from raw text to a trained, evaluated classifier — using well-established NLP and machine learning libraries.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `NLTK` | Tokenization, lemmatization, stop-word corpus |
| `Scikit-learn` | ML classifiers, TF-IDF / BoW, evaluation metrics |
| `Matplotlib / Seaborn` | Data visualization, confusion matrix, distribution plots |
| `Pandas / NumPy` | Dataset loading, manipulation, and statistics |
| `WordCloud` *(optional)* | Visual representation of frequent sentiment words |

---

## 🔬 Methodology

### Steps Performed

```
1. Data Collection & Exploration
   └─► Load dataset, inspect class distribution, identify imbalances

2. Text Preprocessing
   └─► Lowercase → Remove punctuation/HTML → Tokenize → Lemmatize → Remove stopwords

3. Feature Engineering
   ├─► Bag-of-Words (BoW) — word count vectors
   └─► TF-IDF — term frequency–inverse document frequency vectors

4. Model Training
   ├─► Naive Bayes (MultinomialNB) — probabilistic baseline
   ├─► Logistic Regression — linear discriminative classifier
   └─► (Optional) SVM — maximum margin classifier

5. Evaluation
   └─► Accuracy · Precision · Recall · F1-Score · Confusion Matrix

6. Results Visualization
   └─► Confusion matrix heatmap · Score distribution · Word clouds
```

---

## 📁 Project Structure

```
Assignment-3-Sentiment-Analysis/
│
├── notebooks/
│   └── NLP_Assignment_3.ipynb  # Complete implementation with outputs
│
├── data/                        # Dataset files (CSV / text)
├── results/                     # Output plots, metrics, confusion matrices
└── README.md                    # This file
```

---

## 🚀 How to Run

### Option A — Jupyter Notebook (Local)
```bash
cd Assignment-3-Sentiment-Analysis
pip install nltk scikit-learn matplotlib seaborn pandas numpy
jupyter notebook notebooks/NLP_Assignment_3.ipynb
```

### Option B — Google Colab
1. Upload `notebooks/NLP_Assignment_3.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells: **Runtime → Run all**

---

## 📊 Key Outputs

The notebook produces the following outputs:

| Output | Description |
|--------|-------------|
| Class distribution plot | Visualizes sentiment label balance |
| Preprocessing comparison | Before/after text cleaning examples |
| Model accuracy scores | Accuracy per classifier |
| Classification report | Precision, Recall, F1-score per class |
| Confusion matrix | Heatmap of predictions vs ground truth |

---

## 🎓 Key NLP Concepts

| Concept | Explanation |
|---------|-------------|
| **Tokenization** | Splitting text into individual word units (tokens) |
| **Lemmatization** | Reducing words to their base form (`running → run`) |
| **TF-IDF** | Weighs words by importance: high TF but low DF = more informative |
| **Bag-of-Words** | Represents text as word count vectors (ignores word order) |
| **Precision vs Recall** | Precision = correctness of positive predictions; Recall = coverage of actual positives |

---

## 👤 Author

**Muhammad Zohair Hassnain** | Student ID: `22F-3150`

[⬆ Back to Main Repository](../README.md)

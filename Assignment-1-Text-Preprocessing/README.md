<div align="center">

<h1>🔬 Assignment 1 — Customer Reviews Intelligence System</h1>

<p><em>An end-to-end NLP pipeline for analyzing customer feedback using Classical Machine Learning</em></p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-4B8BBE?style=flat-square)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?style=flat-square)](https://gradio.app)

</div>

---

## 📌 Objective

Build a complete, modular NLP system that processes customer reviews and delivers three types of intelligence:

- **Sentiment Analysis** — Classify reviews as Positive, Negative, or Neutral
- **Intent Classification** — Detect user intent: Complaint, Refund Request, Query, or Delivery Feedback
- **Topic Discovery** — Uncover hidden themes from review corpus using unsupervised NMF
- **Interactive UI** — A Gradio-powered web interface for real-time predictions

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `NLTK` | 3.8+ | Tokenization, stop-word removal, corpus utilities |
| `Scikit-learn` | 1.3+ | Naive Bayes, Logistic Regression, TF-IDF, NMF |
| `Gradio` | 4.0+ | Interactive web interface |
| `Matplotlib / Seaborn` | latest | Confusion matrices and evaluation charts |
| `fpdf` | latest | PDF report generation |
| `NumPy / Pandas` | latest | Data manipulation |

---

## 🔬 Methodology

```
Raw Customer Reviews
        │
        ▼
┌─────────────────────────┐
│   Text Preprocessing    │  Lowercase → URL Removal → Roman Urdu
│                         │  Normalization → Tokenization → Stopword Removal
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  TF-IDF Vectorization   │  Transform text into numerical feature vectors
└─────────────────────────┘
        │
     ┌──┴──────────────────┐
     ▼                     ▼
┌──────────────┐   ┌──────────────────────┐
│  Sentiment   │   │  Intent Classifier   │
│  Naive Bayes │   │  Logistic Regression │
└──────────────┘   └──────────────────────┘
     │                     │
     └──────────┬───────────┘
                ▼
        ┌──────────────┐
        │ Topic Model  │  NMF — Unsupervised (3 topics)
        └──────────────┘
                │
                ▼
        ┌──────────────┐
        │  Gradio UI   │  Real-time interactive predictions
        └──────────────┘
```

---

## 📁 Project Structure

```
Assignment-1-Text-Preprocessing/
│
├── src/
│   ├── preprocessing.py      # Text cleaning, tokenization, Roman Urdu normalization
│   ├── sentiment.py          # Multinomial Naive Bayes sentiment classifier
│   ├── intent.py             # Logistic Regression intent classifier
│   ├── topic_modeling.py     # NMF-based unsupervised topic discovery
│   ├── vectorization.py      # TF-IDF feature extraction utilities
│   ├── evaluation.py         # Metrics computation and confusion matrix plots
│   └── utils.py              # Data loading, model save/load, helpers
│
├── interface/
│   └── app.py                # Gradio web application
│
├── notebooks/
│   └── NLP_Assignment_1.ipynb  # Step-by-step code demonstration
│
├── data/
│   ├── raw/                  # Original customer review datasets
│   ├── processed/            # Cleaned and preprocessed data
│   └── augmented/            # Augmented samples for training
│
├── models/                   # Saved trained model files (*.pkl)
├── reports/
│   └── figures/              # Evaluation plots and confusion matrices
│
├── main.py                   # Pipeline orchestrator (run this first)
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
cd Assignment-1-Text-Preprocessing
pip install -r requirements.txt
```

### Step 2 — Train the Models
```bash
python main.py
```
This will load the sample dataset, train all three models, evaluate their performance, and save trained models to `models/`.

### Step 3 — Launch the Gradio Interface
```bash
python interface/app.py
```
Open the URL shown in your terminal (usually `http://127.0.0.1:7860`) to interact with the system in real time.

---

## 📊 Results

| Model | Task | Algorithm | Output Classes |
|-------|------|-----------|----------------|
| Sentiment Classifier | 3-class | Multinomial Naive Bayes | Positive · Negative · Neutral |
| Intent Classifier | 4-class | Logistic Regression | Complaint · Refund · Query · Delivery |
| Topic Modeler | Unsupervised | NMF | 3 hidden topics |

> Confusion matrices and evaluation figures are saved to `reports/figures/` after running `main.py`.

---

## 🎓 Key NLP Concepts

| Concept | Explanation |
|---------|-------------|
| **Why Naive Bayes?** | Probabilistic model based on word frequency — fast and effective for high-dimensional sparse text features |
| **Why TF-IDF?** | Assigns higher weights to rare, informative words while penalizing overly common "noise" words |
| **What is NMF?** | Non-Negative Matrix Factorization decomposes a document-term matrix into latent topic distributions |
| **Roman Urdu Handling** | Custom regex normalization rules (`bht → bohat`, `ny → nai`) for transliterated Urdu text |
| **Sentiment Stop Words** | Words like `not`, `no`, `never` are intentionally kept — they reverse sentiment meaning |

---

## 👤 Author

**Muhammad Zohair Hassnain** | Student ID: `22F-3150`

[⬆ Back to Main Repository](../README.md)

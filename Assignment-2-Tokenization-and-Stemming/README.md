<div align="center">

<h1>🔐 Assignment 2 — Phishing Detection via Deep Learning</h1>

<p><em>A comparative study of classical and neural network models for phishing text classification</em></p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📌 Objective

Perform a rigorous comparative analysis of four machine learning models for detecting phishing content in text:

1. **Logistic Regression** — Classical ML baseline with GridSearchCV tuning
2. **Feedforward Neural Network (FNN)** — Dense architecture with dropout regularization
3. **Recurrent Neural Network (RNN)** — Sequential text modeling with SimpleRNN
4. **Long Short-Term Memory (LSTM)** — Advanced sequential model with memory gates

The goal is to understand the trade-offs between model complexity, accuracy, and computational cost for NLP text classification.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Building and training FNN, RNN, LSTM models |
| `Scikit-learn` | Logistic Regression, TF-IDF vectorization, GridSearchCV |
| `Pandas / NumPy` | Data loading and manipulation |
| `Matplotlib / Seaborn` | ROC curves, confusion matrices, training history plots |

---

## 🔬 Methodology

### Pipeline Overview

```
Raw Phishing Dataset (CSV)
        │
        ▼
┌──────────────────────────┐
│    Text Preprocessing    │  Lowercase → Punctuation removal → Stopword filtering
└──────────────────────────┘
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
TF-IDF Vectors              Padded Sequences
(for LR and FNN)           (for RNN and LSTM)
   │                              │
   ├──► Logistic Regression       ├──► SimpleRNN (32 units)
   └──► FNN (64→32→sigmoid)       └──► LSTM (32 units)
        │                              │
        └──────────────┬───────────────┘
                       ▼
            Model Evaluation & Comparison
            (Accuracy, Precision, Recall, F1, AUC-ROC)
```

### Steps Performed

1. **Data Loading** — Read phishing dataset from CSV, inspect class distribution
2. **Preprocessing** — Clean text: lowercase, remove punctuation, filter stop words
3. **Feature Extraction**:
   - TF-IDF sparse vectors → used by Logistic Regression and FNN
   - Tokenized and padded integer sequences → used by RNN and LSTM
4. **Model Training**:
   - Logistic Regression with `GridSearchCV` (C: [1, 10])
   - FNN: `Dense(64, relu) → Dropout(0.5) → Dense(32, relu) → sigmoid`
   - RNN: `Embedding → SimpleRNN(32, dropout=0.2) → sigmoid`
   - LSTM: `Embedding → LSTM(32, dropout=0.2) → sigmoid`
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC per model
6. **Visualization** — Confusion matrices and ROC curves saved to `outputs/`

---

## 📁 Project Structure

```
Assignment-2-Tokenization-and-Stemming/
│
├── src/
│   ├── preprocessing.py    # Text cleaning and feature extraction (TF-IDF, sequences)
│   ├── models.py           # Model architecture definitions (LR, FNN, RNN, LSTM)
│   ├── training.py         # Training loops for sklearn and Keras models
│   └── evaluation.py       # Metrics computation, confusion matrix, ROC curves
│
├── notebooks/
│   └── NLP_Assignment_2.ipynb  # Full step-by-step walkthrough notebook
│
├── data/
│   └── phishing_dataset.csv    # Sample phishing detection dataset
│
├── outputs/                    # Saved evaluation plots and result tables
├── test_pipeline.py            # End-to-end pipeline runner script
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
cd Assignment-2-Tokenization-and-Stemming
pip install -r requirements.txt
```

### Step 2 — Run the Full Pipeline
```bash
python test_pipeline.py
```
This trains all 4 models, evaluates them, prints a results table, and saves plots to `outputs/`.

### Step 3 — Explore the Notebook
```bash
jupyter notebook notebooks/NLP_Assignment_2.ipynb
```

---

## 📊 Model Comparison

| Model | Architecture | Feature Repr. | Strengths |
|-------|-------------|---------------|-----------|
| **Logistic Regression** | Linear + GridSearchCV | TF-IDF | Fast, interpretable baseline |
| **FNN** | Dense(64)→Dense(32)→Sigmoid | TF-IDF | Captures non-linear patterns |
| **RNN** | Embedding→SimpleRNN(32) | Padded Seq | Sequential text context |
| **LSTM** | Embedding→LSTM(32) | Padded Seq | Long-range dependencies, best accuracy |

> All results including Accuracy, Precision, Recall, F1-score, and AUC are printed to the console and saved in `outputs/`.

---

## 🎓 Key NLP Concepts

| Concept | Explanation |
|---------|-------------|
| **TF-IDF vs Embeddings** | TF-IDF creates sparse bag-of-words; embeddings capture semantic similarity |
| **Why LSTM over RNN?** | LSTM uses memory gates to retain long-range dependencies and avoid vanishing gradient |
| **Dropout Regularization** | Prevents overfitting by randomly deactivating neurons during training |
| **GridSearchCV** | Exhaustive hyperparameter search with cross-validation for classical models |

---

## 👤 Author

**Muhammad Zohair Hassnain** | Student ID: `22F-3150`

[⬆ Back to Main Repository](../README.md)

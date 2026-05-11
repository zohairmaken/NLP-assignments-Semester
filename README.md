<div align="center">

<h1>🧠 NLP Assignments Portfolio</h1>

<h3>Muhammad Zohair Hassnain &nbsp;|&nbsp; 22F-3150</h3>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<p><em>A curated collection of Natural Language Processing assignments covering classical ML, deep learning, and transformer-based approaches — from text preprocessing to explainability analysis.</em></p>

</div>

---

## 📚 Assignment Overview

| # | Assignment | Core Topics | Tech Stack |
|---|-----------|-------------|------------|
| **01** | [🔬 Customer Reviews NLP Pipeline](./Assignment-1-NLP-Pipeline/) | Text Preprocessing · Sentiment Analysis · Intent Classification · Topic Modeling · Gradio UI | NLTK · Scikit-learn · NMF · Gradio |
| **02** | [🔐 Phishing Detection — Deep Learning](./Assignment-2-Phishing-Detection/) | Binary Text Classification · Model Comparison · Neural Networks | TensorFlow · Keras · LSTM · RNN · FNN |
| **03** | [💬 Sentiment Analysis](./Assignment-3-Sentiment-Analysis/) | Sentiment Classification · Feature Engineering · Evaluation Metrics | NLTK · Scikit-learn · Matplotlib |
| **04** | [🤖 Transformer Explainability](./Assignment-4-Transformer-Explainability/) | DistilBERT Fine-tuning · Attention Visualization · SHAP · LIME | HuggingFace · PyTorch · BertViz · SHAP · LIME |

---

## 🗂️ Repository Structure

```
NLP-Assignments/
│
├── Assignment-1-NLP-Pipeline/           # End-to-end NLP pipeline with Gradio UI
│   ├── src/                             # Core modules (preprocessing, sentiment, intent, topic)
│   ├── interface/                       # Gradio web application
│   ├── notebooks/                       # Jupyter demonstration
│   ├── data/                            # Raw, processed, augmented datasets
│   ├── models/                          # Saved trained models (.pkl)
│   ├── reports/                         # Evaluation figures & confusion matrices
│   └── main.py                          # Pipeline orchestrator
│
├── Assignment-2-Phishing-Detection/     # Deep learning text classification
│   ├── src/                             # Model definitions, training, evaluation
│   ├── notebooks/                       # Full walkthrough notebook
│   ├── data/                            # Phishing dataset (CSV)
│   └── outputs/                         # Plots, ROC curves, metrics
│
├── Assignment-3-Sentiment-Analysis/     # Classical sentiment analysis
│   ├── notebooks/                       # Complete implementation notebook
│   ├── data/                            # Dataset files
│   └── results/                         # Output plots and metrics
│
├── Assignment-4-Transformer-Explainability/  # DistilBERT + SHAP + LIME
│   ├── notebooks/                       # Google Colab notebook
│   └── results/                         # Attention maps, SHAP/LIME plots
│
├── assets/                              # Shared assets
├── .gitignore
├── LICENSE
├── requirements.txt                     # Combined dependencies
└── README.md                            # This file
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/NLP-Assignments.git
cd NLP-Assignments

# 2. Install global dependencies
pip install -r requirements.txt

# 3. Navigate to any assignment
cd Assignment-1-NLP-Pipeline
pip install -r requirements.txt
python main.py
```

> 💡 Each assignment has its own `README.md` with detailed setup and run instructions.

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Classical NLP** | NLTK, Scikit-learn, TF-IDF, NMF |
| **Deep Learning** | TensorFlow 2.x, Keras, LSTM, RNN, FNN |
| **Transformers** | HuggingFace Transformers, DistilBERT, PyTorch |
| **Explainability** | SHAP, LIME, BertViz |
| **Visualization** | Matplotlib, Seaborn |
| **UI** | Gradio |
| **Notebooks** | Jupyter Notebook, Google Colab |

---

## 📈 Skills Demonstrated

- ✅ Text preprocessing pipelines (tokenization, normalization, Roman Urdu handling)
- ✅ Classical ML classifiers (Naive Bayes, Logistic Regression, TF-IDF)
- ✅ Deep learning for NLP (FNN, RNN, LSTM with Keras)
- ✅ Transformer fine-tuning with HuggingFace Trainer API
- ✅ Model interpretability (SHAP, LIME, Attention visualization)
- ✅ End-to-end pipeline design with modular code architecture
- ✅ Interactive UI development with Gradio

---

## 👤 Author

**Muhammad Zohair Hassnain**
Student ID: `22F-3150`
Natural Language Processing Course

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

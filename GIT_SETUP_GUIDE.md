# 🚀 Git Setup & Commit Guide

**Repository:** `NLP-Assignments`
**Author:** Muhammad Zohair Hassnain | `22F-3150`

---

## Step 1 — Initialize the Repository

```bash
# Navigate to the repo folder
cd "NLP-Assignments"

# Initialize Git
git init

# Set your identity (first-time setup)
git config user.name "Muhammad Zohair Hassnain"
git config user.email "your.email@example.com"
```

---

## Step 2 — Initial Commit (Root Files)

```bash
git add README.md .gitignore LICENSE requirements.txt GIT_SETUP_GUIDE.md
git commit -m "chore: initialize professional NLP assignments repository

- Add MIT License for Muhammad Zohair Hassnain (22F-3150)
- Add comprehensive .gitignore for Python/Jupyter projects
- Add combined requirements.txt covering all 4 assignments
- Add portfolio-style README with badges and assignment overview table"
```

---

## Step 3 — Assignment 1

```bash
git add Assignment-1-NLP-Pipeline/
git commit -m "feat(a1): add Customer Reviews Intelligence System

- End-to-end NLP pipeline: preprocessing, sentiment, intent, topic modeling
- Multinomial Naive Bayes for 3-class sentiment classification
- Logistic Regression for 4-class intent detection
- NMF unsupervised topic discovery (3 topics)
- Gradio web interface for real-time predictions
- Roman Urdu text normalization support
- Modular src/ architecture with separate evaluation and utils modules"
```

---

## Step 4 — Assignment 2

```bash
git add Assignment-2-Phishing-Detection/
git commit -m "feat(a2): add Phishing Detection via Deep Learning

- Comparative analysis: Logistic Regression, FNN, RNN, and LSTM
- TF-IDF vectors for classical models; padded sequences for neural nets
- GridSearchCV hyperparameter tuning for Logistic Regression
- Keras FNN: Dense(64) -> Dropout(0.5) -> Dense(32) -> Sigmoid
- RNN and LSTM with Embedding layer and dropout regularization
- ROC curves, confusion matrices saved to outputs/
- Phishing dataset included in data/"
```

---

## Step 5 — Assignment 3

```bash
git add Assignment-3-Sentiment-Analysis/
git commit -m "feat(a3): add Sentiment Analysis notebook

- Classical NLP sentiment classification pipeline
- Text preprocessing: tokenization, lemmatization, stop-word removal
- Feature extraction: Bag-of-Words and TF-IDF representations
- Model evaluation: accuracy, precision, recall, F1-score
- Confusion matrix and score distribution visualizations"
```

---

## Step 6 — Assignment 4

```bash
git add Assignment-4-Transformer-Explainability/
git commit -m "feat(a4): add Transformer Explainability Analysis

- Fine-tune DistilBERT on Amazon Polarity dataset (10,000 samples)
- Achieve ~91% accuracy and F1-score on binary sentiment classification
- Attention visualization using BertViz (Layer 0 and Layer 5)
- SHAP global feature importance analysis across 20 samples
- LIME local interpretable explanations per prediction
- Comparative analysis: SHAP vs LIME (stability, faithfulness, speed)
- Error analysis: sarcasm and negation misclassification patterns
- Developed on Google Colab with Tesla T4 GPU"
```

---

## Step 7 — Push to GitHub

```bash
# Add your GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/NLP-Assignments.git

# Rename branch to main
git branch -M main

# Push all commits
git push -u origin main
```

---

## 📋 GitHub Repository Settings

### Repository Description
```
🧠 NLP Assignments — Classical ML, Deep Learning & Transformer-based NLP | NLTK · TensorFlow · HuggingFace · SHAP · LIME | Muhammad Zohair Hassnain (22F-3150)
```

### Topics / Tags (add in GitHub Settings)
```
nlp, natural-language-processing, machine-learning, deep-learning,
transformers, sentiment-analysis, text-classification, python,
jupyter-notebook, huggingface, distilbert, lstm, gradio, shap, lime,
university-assignment
```

### GitHub About Section
```
🧠 Four NLP assignments: text preprocessing pipelines, phishing detection
with LSTM/RNN, sentiment analysis, and DistilBERT fine-tuning with SHAP & LIME explainability.
```

---

## 📌 Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) for all future commits:

| Prefix | Use For |
|--------|---------|
| `feat:` | New feature or assignment |
| `fix:` | Bug fix |
| `docs:` | Documentation updates |
| `chore:` | Config, dependencies, setup |
| `refactor:` | Code restructuring (no logic change) |
| `test:` | Adding or updating tests |

**Example:**
```bash
git commit -m "docs(a1): improve README with methodology flowchart"
git commit -m "fix(a2): resolve TF-IDF shape mismatch in FNN input"
```

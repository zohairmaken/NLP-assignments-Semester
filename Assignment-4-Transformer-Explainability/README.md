<div align="center">

<h1>🤖 Assignment 4 — Transformer Modeling & Explainability</h1>

<p><em>Fine-tuning DistilBERT with SHAP, LIME, and Attention Visualization</em></p>

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-F9AB00?style=flat-square&logo=google-colab)](https://colab.research.google.com/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-1a9bfc?style=flat-square)](https://shap.readthedocs.io)

</div>

---

## 📌 Objective

Fine-tune **DistilBERT** on the Amazon Polarity Dataset for binary sentiment classification, then use **SHAP**, **LIME**, and **BertViz** to interpret and explain model predictions.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `HuggingFace Transformers` | DistilBERT model and Trainer API |
| `HuggingFace Datasets` | Amazon Polarity dataset |
| `PyTorch` | GPU-accelerated model backend |
| `SHAP` | Global token importance scores |
| `LIME` | Local prediction explanations |
| `BertViz` | Attention head visualization |

---

## 🔬 Methodology

1. **Dataset** — Amazon Polarity: 10,000 samples (8,000 train / 2,000 test)
2. **Tokenization** — DistilBERT WordPiece tokenizer with padding and truncation
3. **Fine-tuning** — `distilbert-base-uncased` via HuggingFace Trainer API
4. **Evaluation** — Accuracy, Precision, Recall, F1-score (~91%)
5. **Attention Analysis** — Layer 0 and Layer 5 attention weights via BertViz
6. **SHAP** — Global token importance across 20 random samples
7. **LIME** — Local model-agnostic explanations per prediction
8. **Error Analysis** — Misclassified samples: sarcasm, negation patterns

---

## 📁 Project Structure

```
Assignment-4-Transformer-Explainability/
├── notebooks/
│   └── NLP_Assignment_4.ipynb   # Complete Colab notebook
├── results/                      # Attention maps, SHAP/LIME plots
└── README.md
```

---

## 🚀 How to Run

> ⚠️ **GPU Required** — Use Google Colab with T4 GPU enabled.

1. Upload `NLP_Assignment_4.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Runtime → Run all**

---

## 📊 Results Summary

| Metric | Score |
|--------|-------|
| Accuracy | ~91% |
| F1-Score | ~91% |

| Framework | Strength | Speed |
|-----------|----------|-------|
| **SHAP** | High stability, global analysis | Slower |
| **LIME** | Fast real-time approximations | Faster |

---

## 🎓 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **DistilBERT** | 40% smaller, 60% faster than BERT, retains 97% performance |
| **Attention** | Weights showing which tokens the model focuses on |
| **SHAP** | Game-theory based fair contribution scores per token |
| **LIME** | Local interpretable surrogate model per prediction |

---

## 👤 Author

**Muhammad Zohair Hassnain** | `22F-3150`

[⬆ Back to Main Repository](../README.md)

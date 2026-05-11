# 🚩 Git Portfolio Setup Guide

This guide provides professional Git instructions to initialize, commit, and push this NLP portfolio to GitHub with a clean, industry-standard commit history.

---

## 🛠️ Phase 1: Initialization

Open your terminal in the root directory (`NLP-Assignments`) and run:

```bash
# 1. Initialize a new Git repository
git init

# 2. Add all files (respecting .gitignore)
git add .

# 3. Initial commit
git commit -m "chore: initial repository structure setup"
```

---

## 📝 Phase 2: Professional Commit History

Instead of a single giant commit, it is recommended to use **Atomic Commits** with **Conventional Commits** prefixes (`feat:`, `docs:`, `chore:`, `refactor:`).

### Step-by-Step Commit Suggestions:

1. **Setup Project 1**
   ```bash
   git add Assignment-1-Text-Preprocessing/
   git commit -m "feat(a1): implement text preprocessing and intent classification pipeline"
   ```

2. **Setup Project 2**
   ```bash
   git add Assignment-2-Tokenization-and-Stemming/
   git commit -m "feat(a2): add phishing detection study with LSTM/RNN/FNN comparisons"
   ```

3. **Setup Project 3**
   ```bash
   git add Assignment-3-Sentiment-Analysis/
   git commit -m "feat(a3): implement classical sentiment analysis with BoW/TF-IDF/Word2Vec"
   ```

4. **Setup Project 4**
   ```bash
   git add Assignment-4-Text-Classification/
   git commit -m "feat(a4): fine-tune DistilBERT with SHAP/LIME explainability"
   ```

5. **Finalize Documentation**
   ```bash
   git add README.md requirements.txt .gitignore LICENSE
   git commit -m "docs: finalize portfolio documentation and environment configuration"
   ```

---

## 🚀 Phase 3: Pushing to GitHub

1. Create a **New Repository** on [GitHub](https://github.com/new) (keep it empty).
2. Copy your repository URL (e.g., `https://github.com/username/NLP-Assignments.git`).
3. Run the following:

```bash
# Link local repo to GitHub
git remote add origin https://github.com/username/NLP-Assignments.git

# Rename main branch (industry standard)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## 💡 Pro Tips for a Great Portfolio

- **Meaningful Messages**: Use the `feat(scope): message` format.
- **No Large Data**: Ensure large datasets are in `.gitignore`.
- **Verified Commits**: If possible, sign your commits with GPG.
- **Activity Graph**: Frequent, smaller commits look better on your GitHub profile than one large push.

---

[⬆ Back to Main README](README.md)

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from preprocessing import load_and_preprocess
from models import split_data, get_tfidf_vectors, get_sequences, build_logistic_regression, build_fnn, build_rnn, build_lstm
from training import train_sklearn_model, train_keras_model
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curves

print("Loading data...")
df = load_and_preprocess("data/phishing_dataset.csv")
print(f"Data loaded. Shape: {df.shape}")
print(df['label'].value_counts(normalize=True))

print("Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

# TF-IDF for LR and FNN
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf = get_tfidf_vectors(X_train, X_val, X_test)

# Sequences for RNN and LSTM
X_train_seq, X_val_seq, X_test_seq, tokenizer = get_sequences(X_train, X_val, X_test)
print("Vectorization complete.")

# 1. Logistic Regression
print("Training Logistic Regression...")
lr_base = build_logistic_regression()
lr_model, time_lr = train_sklearn_model(lr_base, X_train_tfidf, y_train, param_grid={'C': [1, 10]})

# 2. FNN
print("Training FNN...")
fnn_base = build_fnn(X_train_tfidf.shape[1])
fnn_model, history_fnn, time_fnn = train_keras_model(fnn_base, X_train_tfidf.toarray(), y_train, X_val_tfidf.toarray(), y_val)

# 3. RNN
print("Training RNN...")
rnn_base = build_rnn(5000, 32, 50)
rnn_model, history_rnn, time_rnn = train_keras_model(rnn_base, X_train_seq, y_train, X_val_seq, y_val)

# 4. LSTM
print("Training LSTM...")
lstm_base = build_lstm(5000, 32, 50)
lstm_model, history_lstm, time_lstm = train_keras_model(lstm_base, X_train_seq, y_train, X_val_seq, y_val)

print("All models trained.")

results = []
roc_data = []

# LR
probs_lr = lr_model.predict_proba(X_test_tfidf)[:, 1]
preds_lr = lr_model.predict(X_test_tfidf)
results.append(evaluate_model(y_test, preds_lr, probs_lr, "Logistic Regression"))
roc_data.append((y_test, probs_lr, "LR"))

# FNN
probs_fnn = fnn_model.predict(X_test_tfidf.toarray()).flatten()
preds_fnn = (probs_fnn > 0.5).astype(int)
results.append(evaluate_model(y_test, preds_fnn, probs_fnn, "FNN"))
roc_data.append((y_test, probs_fnn, "FNN"))

# RNN
probs_rnn = rnn_model.predict(X_test_seq).flatten()
preds_rnn = (probs_rnn > 0.5).astype(int)
results.append(evaluate_model(y_test, preds_rnn, probs_rnn, "RNN"))
roc_data.append((y_test, probs_rnn, "RNN"))

# LSTM
probs_lstm = lstm_model.predict(X_test_seq).flatten()
preds_lstm = (probs_lstm > 0.5).astype(int)
results.append(evaluate_model(y_test, preds_lstm, probs_lstm, "LSTM"))
roc_data.append((y_test, probs_lstm, "LSTM"))

results_df = pd.DataFrame(results)
print("\n--- RESULTS ---")
print(results_df)

print("\nScript execution complete. Notebook logic verified.")

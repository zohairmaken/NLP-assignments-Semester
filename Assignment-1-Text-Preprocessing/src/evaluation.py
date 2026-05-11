"""
evaluation.py
-------------
Measures how well our models are performing.
Calculates:
- Precision: "Of all items predicted as Positive, how many were actually Positive?"
- Recall: "Of all actually Positive items, how many did we correctly find?"
- F1-Score: The harmonic mean of Precision and Recall.
- Confusion Matrix: Visualizes correct vs. incorrect predictions.

Why this matters in viva:
  - In customer reviews, Recall is often more important for 'Complaints'. 
    We don't want to miss a single angry customer!
  - F1-score is best when classes are imbalanced (e.g., way more positive reviews than negative).
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(y_true, y_pred, model_name="Model", save_path=None):
    """
    Prints a detailed report and saves a confusion matrix plot.
    """
    print(f"\nEvaluation for {model_name}:")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
    
    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion Matrix saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def get_performance_summary(y_true, y_pred):
    """Returns basic metrics for comparison tables."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

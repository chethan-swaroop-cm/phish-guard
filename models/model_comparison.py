import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set local Matplotlib config dir to avoid permission warnings
mpl_dir = PROJECT_ROOT / "data" / ".mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/CLI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model_performance(y_true, y_pred):
    """Computes standard evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

def train_and_compare_models(csv_path: Path, task_name: str = "Spam vs Ham", test_size: float = 0.25, random_state: int = 42):
    """
    Trains and benchmarks Naive Bayes vs Logistic Regression
    using TF-IDF Vectorization on the given labelled dataset.
    """
    # 1. Load Dataset
    df = pd.read_csv(csv_path)
    X = df["text"].tolist()
    y = df["label"].tolist()

    # 2. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4. Train Naive Bayes
    nb_clf = MultinomialNB(alpha=0.1)
    nb_clf.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_clf.predict(X_test_tfidf)
    nb_metrics = evaluate_model_performance(y_test, y_pred_nb)

    # 5. Train Logistic Regression
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    lr_clf.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_clf.predict(X_test_tfidf)
    lr_metrics = evaluate_model_performance(y_test, y_pred_lr)

    # 6. Generate Side-by-Side Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_nb = np.array(nb_metrics["confusion_matrix"])
    cm_lr = np.array(lr_metrics["confusion_matrix"])

    # Labels for classes
    labels = ["Safe / Ham (0)", "Threat / Malicious (1)"]

    sns.heatmap(
        cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
        xticklabels=labels, yticklabels=labels, cbar=False
    )
    axes[0].set_title(f"Naive Bayes Confusion Matrix\n({task_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1],
        xticklabels=labels, yticklabels=labels, cbar=False
    )
    axes[1].set_title(f"Logistic Regression Confusion Matrix\n({task_name})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    output_img_path = PROJECT_ROOT / "data" / f"confusion_matrix_{task_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_img_path, dpi=200)
    plt.close()

    # 7. Comparison Summary Table
    comparison_df = pd.DataFrame([
        {
            "Algorithm": "Multinomial Naive Bayes",
            "Accuracy": nb_metrics["accuracy"],
            "Precision": nb_metrics["precision"],
            "Recall": nb_metrics["recall"],
            "F1-Score": nb_metrics["f1_score"]
        },
        {
            "Algorithm": "Logistic Regression",
            "Accuracy": lr_metrics["accuracy"],
            "Precision": lr_metrics["precision"],
            "Recall": lr_metrics["recall"],
            "F1-Score": lr_metrics["f1_score"]
        }
    ])

    return {
        "task_name": task_name,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "comparison_table": comparison_df,
        "naive_bayes": nb_metrics,
        "logistic_regression": lr_metrics,
        "confusion_matrix_image": str(output_img_path),
        "vectorizer": tfidf,
        "nb_model": nb_clf,
        "lr_model": lr_clf
    }

if __name__ == "__main__":
    print("=" * 60)
    print("TF-IDF MODEL TRAINING & COMPARISON BENCHMARK")
    print("=" * 60)

    # Benchmark 1: Spam vs Ham
    spam_csv = PROJECT_ROOT / "data" / "spam_dataset.csv"
    res_spam = train_and_compare_models(spam_csv, task_name="Spam Detection")
    print("\n--- [TASK 1: SPAM DETECTION] ---")
    print(res_spam["comparison_table"].to_string(index=False))
    print(f"Confusion Matrix saved to: {res_spam['confusion_matrix_image']}")

    # Benchmark 2: Phishing vs Legitimate
    phish_csv = PROJECT_ROOT / "data" / "phishing_dataset.csv"
    res_phish = train_and_compare_models(phish_csv, task_name="Email Phishing Detection")
    print("\n--- [TASK 2: EMAIL PHISHING DETECTION] ---")
    print(res_phish["comparison_table"].to_string(index=False))
    print(f"Confusion Matrix saved to: {res_phish['confusion_matrix_image']}")

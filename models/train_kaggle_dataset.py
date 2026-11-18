import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import os

# Set local Matplotlib config dir
mpl_dir = PROJECT_ROOT / "data" / ".mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist()
    }

def train_and_benchmark_kaggle_dataset(csv_path=None, test_size=0.20, random_state=42):
    path = csv_path or (PROJECT_ROOT / "data" / "kaggle_phishing_dataset.csv")
    df = pd.read_csv(path)
    
    # Target column is the last column (Result / class)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    
    # Map label: -1 (Legitimate) -> 0, 1 (Phishing) -> 1
    y = np.where(y_raw == -1, 0, 1)

    # 80:20 Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_res = evaluate_metrics(y_test, y_pred_lr)

    # 2. Gaussian Naive Bayes (for continuous/tabular features)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    nb_res = evaluate_metrics(y_test, y_pred_nb)

    # 3. Random Forest (Top Performer on tabular URL features)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_res = evaluate_metrics(y_test, y_pred_rf)

    # Save Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["Legitimate (0)", "Phishing (1)"]

    sns.heatmap(np.array(nb_res["confusion_matrix"]), annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=labels, yticklabels=labels)
    axes[0].set_title(f"Naive Bayes (Acc: {nb_res['accuracy']*100:.1f}%)", fontweight='bold')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(np.array(lr_res["confusion_matrix"]), annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=labels, yticklabels=labels)
    axes[1].set_title(f"Logistic Regression (Acc: {lr_res['accuracy']*100:.1f}%)", fontweight='bold')
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    sns.heatmap(np.array(rf_res["confusion_matrix"]), annot=True, fmt='d', cmap='Purples', ax=axes[2], xticklabels=labels, yticklabels=labels)
    axes[2].set_title(f"Random Forest (Acc: {rf_res['accuracy']*100:.1f}%)", fontweight='bold')
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    plt.tight_layout()
    output_img = PROJECT_ROOT / "data" / "confusion_matrix_kaggle_11054.png"
    plt.savefig(output_img, dpi=200)
    plt.close()

    summary_df = pd.DataFrame([
        {"Algorithm": "Gaussian Naive Bayes", "Accuracy": nb_res["accuracy"], "Precision": nb_res["precision"], "Recall": nb_res["recall"], "F1-Score": nb_res["f1_score"]},
        {"Algorithm": "Logistic Regression", "Accuracy": lr_res["accuracy"], "Precision": lr_res["precision"], "Recall": lr_res["recall"], "F1-Score": lr_res["f1_score"]},
        {"Algorithm": "Random Forest", "Accuracy": rf_res["accuracy"], "Precision": rf_res["precision"], "Recall": rf_res["recall"], "F1-Score": rf_res["f1_score"]}
    ])

    return {
        "dataset_name": "Kaggle / UCI Phishing Websites (11,054 records)",
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "features_count": X.shape[1],
        "summary_table": summary_df,
        "naive_bayes": nb_res,
        "logistic_regression": lr_res,
        "random_forest": rf_res,
        "confusion_matrix_img": str(output_img),
        "rf_model": rf
    }

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING ON KAGGLE PHISHING URL DATASET (11,054 RECORDS, 80:20 SPLIT)")
    print("=" * 60)
    res = train_and_benchmark_kaggle_dataset()
    print(f"\nTotal Records: {res['total_records']} | Train (80%): {res['train_records']} | Test (20%): {res['test_records']}")
    print(f"Features: {res['features_count']}")
    print("\nBENCHMARK RESULTS:")
    print(res["summary_table"].to_string(index=False))
    print(f"\nConfusion Matrix Heatmap saved to: {res['confusion_matrix_img']}")

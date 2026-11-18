import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

class TextClassificationPipeline:
    """
    Reusable Text Classification Pipeline wrapping TF-IDF vectorization
    and a configurable scikit-learn estimator (e.g. MultinomialNB or LogisticRegression).
    Used interchangeably for Spam and Email Phishing tasks.
    """
    def __init__(self, classifier, name: str = "TextClassifier", max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.name = name
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            sublinear_tf=True
        )
        self.classifier = classifier
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.classifier)
        ])
        self.is_fitted = False
        
    def fit(self, texts: List[str], labels: List[int]):
        """Trains the TF-IDF vectorizer and downstream classifier."""
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self
        
    def predict(self, texts: List[str]) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        return self.pipeline.predict(texts)
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns probability distribution [P(class=0), P(class=1)]."""
        if hasattr(self.pipeline.named_steps['clf'], "predict_proba"):
            return self.pipeline.predict_proba(texts)
        elif hasattr(self.pipeline.named_steps['clf'], "decision_function"):
            decision = self.pipeline.decision_function(texts)
            prob_1 = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - prob_1, prob_1]).T
        else:
            preds = self.predict(texts)
            return np.vstack([1 - preds, preds]).T

    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """Calculates standard classification evaluation metrics."""
        preds = self.predict(test_texts)
        acc = accuracy_score(test_labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary', zero_division=0)
        report = classification_report(test_labels, preds, output_dict=True, zero_division=0)
        return {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "detailed_report": report
        }

    def save(self, filepath: Path):
        """Serializes pipeline to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, str(filepath))

    def load(self, filepath: Path):
        """Loads serialized pipeline from disk."""
        self.pipeline = joblib.load(str(filepath))
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.classifier = self.pipeline.named_steps['clf']
        self.is_fitted = True
        return self

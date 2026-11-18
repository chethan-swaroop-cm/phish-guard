from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from models.base_pipeline import TextClassificationPipeline
import config

class PhishingDetector:
    """
    Email phishing detection using TF-IDF + Logistic Regression.
    Includes feature-based Explainable AI (XAI) using linear coefficients and TF-IDF activation.
    """
    def __init__(self, c_param: float = 1.0):
        self.pipeline = TextClassificationPipeline(
            classifier=LogisticRegression(C=c_param, max_iter=1000, random_state=42),
            name="PhishingDetector_LogisticRegression",
            max_features=5000,
            ngram_range=(1, 2)
        )

    def fit(self, texts: List[str], labels: List[int]):
        self.pipeline.fit(texts, labels)
        return self

    def predict_phishing(self, text: str) -> Dict[str, Any]:
        """Predicts phishing probability and classification."""
        proba = self.pipeline.predict_proba([text])[0]
        phish_prob = float(proba[1])
        is_phishing = bool(phish_prob >= config.PHISHING_THRESHOLD)
        return {
            "is_phishing": is_phishing,
            "phishing_probability": phish_prob,
            "safe_probability": float(proba[0])
        }

    def explain(self, text: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Explainable AI for Logistic Regression:
        Calculates feature contribution: w_i * tfidf_i for active vocabulary tokens.
        Positive values indicate keywords driving Phishing verdict.
        Negative values indicate keywords driving Safe/Legitimate verdict.
        """
        if not self.pipeline.is_fitted:
            return {"phishing_triggers": [], "safe_indicators": []}

        vectorizer = self.pipeline.vectorizer
        clf: LogisticRegression = self.pipeline.classifier
        
        tfidf_vec = vectorizer.transform([text]).toarray()[0]
        active_indices = np.where(tfidf_vec > 0)[0]
        
        if len(active_indices) == 0:
            return {"phishing_triggers": [], "safe_indicators": []}

        # Logistic regression coef has shape (1, n_features)
        weights = clf.coef_[0]
        contributions = tfidf_vec * weights
        
        feature_names = vectorizer.get_feature_names_out()
        
        scored = [(feature_names[i], float(contributions[i])) for i in active_indices]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        phishing_triggers = [
            {"token": token, "weight": round(score, 4)}
            for token, score in scored if score > 0
        ][:top_k]
        
        safe_indicators = [
            {"token": token, "weight": round(abs(score), 4)}
            for token, score in reversed(scored) if score < 0
        ][:top_k]
        
        return {
            "phishing_triggers": phishing_triggers,
            "safe_indicators": safe_indicators
        }

    def save(self, path=None):
        self.pipeline.save(path or config.PHISHING_MODEL_PATH)

    def load(self, path=None):
        self.pipeline.load(path or config.PHISHING_MODEL_PATH)
        return self

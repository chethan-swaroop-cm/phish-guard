from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from models.base_pipeline import TextClassificationPipeline
import config

class SpamDetector:
    """
    Spam detection using TF-IDF + Multinomial Naive Bayes.
    Includes feature-based Explainable AI (XAI) using log-likelihood ratios.
    """
    def __init__(self, alpha: float = 0.1):
        self.pipeline = TextClassificationPipeline(
            classifier=MultinomialNB(alpha=alpha),
            name="SpamDetector_NaiveBayes",
            max_features=5000,
            ngram_range=(1, 2)
        )

    def fit(self, texts: List[str], labels: List[int]):
        self.pipeline.fit(texts, labels)
        return self

    def predict_spam(self, text: str) -> Dict[str, Any]:
        """Predicts spam probability and binary flag."""
        proba = self.pipeline.predict_proba([text])[0]
        spam_prob = float(proba[1])
        is_spam = bool(spam_prob >= config.SPAM_THRESHOLD)
        return {
            "is_spam": is_spam,
            "spam_probability": spam_prob,
            "ham_probability": float(proba[0])
        }

    def explain(self, text: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Explainable AI for Naive Bayes:
        Extracts active n-grams with highest log-likelihood difference:
        log P(w|Spam) - log P(w|Ham) weighted by TF-IDF presence.
        """
        if not self.pipeline.is_fitted:
            return {"spam_triggers": [], "ham_indicators": []}

        vectorizer = self.pipeline.vectorizer
        clf: MultinomialNB = self.pipeline.classifier
        
        # Transform single document
        tfidf_vec = vectorizer.transform([text]).toarray()[0]
        active_indices = np.where(tfidf_vec > 0)[0]
        
        if len(active_indices) == 0:
            return {"spam_triggers": [], "ham_indicators": []}

        # MultinomialNB feature log prob: shape (n_classes, n_features)
        # Class 1 is Spam, Class 0 is Ham
        log_prob_spam = clf.feature_log_prob_[1]
        log_prob_ham = clf.feature_log_prob_[0]
        log_diff = log_prob_spam - log_prob_ham
        
        # Weight by TF-IDF value in current message
        weighted_scores = tfidf_vec * log_diff
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Sort active features
        active_scores = [(feature_names[i], float(weighted_scores[i])) for i in active_indices]
        active_scores.sort(key=lambda x: x[1], reverse=True)
        
        spam_triggers = [{"token": token, "weight": round(score, 4)} for token, score in active_scores if score > 0][:top_k]
        ham_indicators = [{"token": token, "weight": round(abs(score), 4)} for token, score in reversed(active_scores) if score < 0][:top_k]
        
        return {
            "spam_triggers": spam_triggers,
            "ham_indicators": ham_indicators
        }

    def save(self, path=None):
        self.pipeline.save(path or config.SPAM_MODEL_PATH)

    def load(self, path=None):
        self.pipeline.load(path or config.SPAM_MODEL_PATH)
        return self

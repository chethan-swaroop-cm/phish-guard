from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import config
from utils.feature_extraction import (
    extract_url_feature_vector,
    extract_url_feature_dict,
    explain_url_heuristics,
    FEATURE_NAMES
)

class URLDetector:
    """
    URL Threat Detection using Lexical/Structural Feature Extraction + ML Classifier
    combined with heuristic rule-based safety overrides.
    """
    def __init__(self, n_estimators: int = 100):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, url_vectors: List[List[float]], labels: List[int]):
        """Trains random forest on tabular URL feature vectors."""
        X = np.array(url_vectors)
        y = np.array(labels)
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict_url(self, url: str) -> Dict[str, Any]:
        """
        Analyzes a single URL through both ML classification
        and heuristic rule-based feature verification.
        """
        feature_dict = extract_url_feature_dict(url)
        feature_vec = extract_url_feature_vector(url)
        heuristics_warnings = explain_url_heuristics(url)
        
        # ML model prediction
        if self.is_fitted:
            proba = self.classifier.predict_proba([feature_vec])[0]
            ml_risk = float(proba[1])
        else:
            # Fallback heuristic score based on flagged anomalies
            anomaly_count = len(heuristics_warnings)
            ml_risk = min(1.0, anomaly_count * 0.25)
            
        # Hard security overrides for severe threats
        # e.g., raw IP address with login keywords or @ sign obfuscation
        final_risk = ml_risk
        if feature_dict["is_ip"] == 1:
            final_risk = max(final_risk, 0.85)
        if feature_dict["num_at_symbols"] > 0:
            final_risk = max(final_risk, 0.80)
        if feature_dict["has_https_in_domain"] == 1:
            final_risk = max(final_risk, 0.75)
            
        is_suspicious = bool(final_risk >= config.URL_RISK_THRESHOLD)
        
        return {
            "url": url,
            "domain": feature_dict.get("domain_length", ""),
            "risk_score": round(final_risk, 4),
            "is_suspicious": is_suspicious,
            "reasons": heuristics_warnings,
            "features": feature_dict
        }

    def evaluate(self, test_vectors: List[List[float]], test_labels: List[int]) -> Dict[str, Any]:
        preds = self.classifier.predict(test_vectors)
        acc = accuracy_score(test_labels, preds)
        report = classification_report(test_labels, preds, output_dict=True, zero_division=0)
        return {
            "accuracy": float(acc),
            "report": report
        }

    def save(self, path=None):
        target = path or config.URL_MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, str(target))

    def load(self, path=None):
        target = path or config.URL_MODEL_PATH
        self.classifier = joblib.load(str(target))
        self.is_fitted = True
        return self

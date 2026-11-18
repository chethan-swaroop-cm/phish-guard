import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
DATABASE_PATH = DATABASE_DIR / "phish_guard.db"

# Ensure directories exist
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifact Paths
SPAM_MODEL_PATH = SAVED_MODELS_DIR / "spam_model.pkl"
PHISHING_MODEL_PATH = SAVED_MODELS_DIR / "phishing_model.pkl"
URL_MODEL_PATH = SAVED_MODELS_DIR / "url_model.pkl"

# Classification Thresholds
SPAM_THRESHOLD = 0.50
PHISHING_THRESHOLD = 0.50
URL_RISK_THRESHOLD = 0.50

# Threat Scoring Weights (Composite Risk)
# Phishing and Malicious URLs pose higher operational risks than plain spam
WEIGHT_PHISHING = 0.45
WEIGHT_URL = 0.40
WEIGHT_SPAM = 0.15

# Suspicious TLDs commonly abused in phishing/spam campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "click", "fit", "work",
    "loan", "date", "racing", "kim", "country", "science", "party", "cricket",
    "faith", "zip", "mov", "su", "icu", "bar", "rest", "cam"
}

# Known URL shorteners that conceal the actual destination
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorturl.at", "tiny.cc", "rebrand.ly"
}

# High-risk brand and action keywords targeted in phishing URLs
SUSPICIOUS_URL_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "banking", "update", "confirm", "billing", "authenticate",
    "password", "credential", "recover", "wallet", "support", "service",
    "portal", "token", "validate", "suspended", "notice", "alert"
]

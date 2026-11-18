# 🛡️ PhishGuard AI: Unified Email & URL Threat Defense

A modular, multi-vector threat detection and Explainable AI (XAI) platform designed to classify spam, detect credential phishing, extract and analyze embedded URLs, and deliver unified threat intelligence.

---

## 🏗️ Architecture & Technology Stack

| Module | Technology | Role & Details |
| :--- | :--- | :--- |
| **Frontend** | **Streamlit** | Interactive UI with real-time threat dashboards, link inspection tables, and XAI tabs. |
| **Backend** | **Python 3.10+** | Core modular pipeline, scoring algorithms, and orchestration. |
| **Spam Detection** | **TF-IDF + Naive Bayes** | Multinomial Naive Bayes text classifier trained on bag-of-words / n-grams. |
| **Email Phishing** | **TF-IDF + Logistic Regression** | Linear classifier measuring credential solicitation and urgency cues. |
| **URL Threat Detection** | **URL Features + ML** | 20+ lexical, structural, and behavioral features + Random Forest classifier. |
| **Data Processing** | **Pandas & NumPy** | Feature vectors, tabular transformations, and dataset curation. |
| **Model Evaluation** | **Scikit-learn** | Precision, recall, F1, accuracy, and confusion matrices. |
| **Explainable AI (XAI)** | **Feature Attributions** | Token weight inspection (NB log-odds, LR coefficients) & heuristic warnings. |
| **Storage & Audit** | **SQLite** | Relational scan audit log (`phish_guard.db`) tracking emails and extracted URLs. |

---

## 🌟 What Makes This Project Stand Out?

### 1. 🔗 Embedded URL Extraction & Multi-Link Analysis
- Automatically extracts **all links** found within raw text or HTML (`<a href="...">`, standard `https://`, defanged `hxxp://`, `[.]`, etc.).
- If an email contains five links, it does not stop at the first: **all five are parsed, feature-engineered, and independently classified**.
- Evaluates raw IP hosts, subdomain depth, high-abuse TLDs, URL shorteners, entropy, and non-standard ports.

### 2. 🎯 Combined Multi-Vector Threat Report
- Attackers frequently craft clean, polite text with a single weaponized link (e.g., invoice fraud).
- Synthesizes multiple vectors into a **Composite Risk Score (0–100%)**:
  $$\text{Composite Risk} = w_{\text{phish}} \cdot P_{\text{phish}} + w_{\text{url}} \cdot \max(P_{\text{url}}) + w_{\text{spam}} \cdot P_{\text{spam}}$$
- Enforces **hard security overrides**: if any embedded link is weaponized or suspicious, the message is immediately escalated to **CRITICAL**, protecting against deceptive lure emails.

### 3. 💡 Explainable AI (XAI)
- Eliminates the "black box" problem:
  - **Phishing**: Reveals exact positive tokens driving phishing classifications ($w_i \cdot x_i$).
  - **Spam**: Dissects log-likelihood ratios ($\log P(w|\text{Spam}) - \log P(w|\text{Ham})$).
  - **URLs**: Generates plain-English security flags (e.g., *"Host uses raw IP address instead of domain"*, *"Contains 4-level subdomain impersonation"*, *"Suspicious .xyz TLD"*).

---

## 📁 Project Structure

```
phish-guard/
├── app.py                     # Streamlit web application & dashboards
├── config.py                  # Thresholds, scoring weights, TLDs, paths
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
├── database/
│   ├── __init__.py
│   ├── db_manager.py          # SQLite schema, scan history, analytics
│   └── phish_guard.db         # Persistent SQLite database (auto-generated)
├── models/
│   ├── __init__.py
│   ├── base_pipeline.py       # Reusable TF-IDF + Estimator pipeline
│   ├── spam_detector.py       # Naive Bayes spam detector & XAI
│   ├── phishing_detector.py   # Logistic Regression phishing detector & XAI
│   ├── url_detector.py        # 20-feature URL classifier & heuristic rules
│   ├── trainer.py             # Model training & seed generation script
│   └── saved_models/          # Serialized .pkl models (auto-generated)
└── utils/
    ├── __init__.py
    ├── url_extractor.py       # Regex & HTML link parser + normalizer
    ├── feature_extraction.py  # 20 lexical/structural URL features & entropy
    ├── explainability.py      # Unified XAI dossier builder
    └── threat_aggregator.py   # Multi-vector decision matrix & executive reporting
```

---

## 🚀 Quickstart & Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```

*Note: On first launch, the application automatically initializes the SQLite database and trains/caches the baseline models if saved artifacts are not yet present.*

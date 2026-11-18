import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import config
from database.db_manager import (
    init_db, save_scan, get_recent_scans, get_scan_details, get_system_stats, clear_all_history
)
from utils.url_extractor import extract_urls
from utils.feature_extraction import extract_url_feature_dict
from utils.threat_aggregator import synthesize_threat_report
from models.spam_detector import SpamDetector
from models.phishing_detector import PhishingDetector
from models.url_detector import URLDetector
from models.trainer import train_and_save_all_models
from models.model_comparison import train_and_compare_models

# Page Config
st.set_page_config(
    page_title="PhishGuard AI - Unified Threat & Phishing Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DB and Models
@st.cache_resource
def load_models():
    init_db()
    # Check if models exist, if not, train default seeds
    if not (config.SPAM_MODEL_PATH.exists() and config.PHISHING_MODEL_PATH.exists() and config.URL_MODEL_PATH.exists()):
        train_and_save_all_models()
        
    spam_det = SpamDetector().load(config.SPAM_MODEL_PATH)
    phish_det = PhishingDetector().load(config.PHISHING_MODEL_PATH)
    url_det = URLDetector().load(config.URL_MODEL_PATH)
    return spam_det, phish_det, url_det

try:
    spam_detector, phishing_detector, url_detector = load_models()
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# Sample Email Templates for Quick Testing
SAMPLE_EMAILS = {
    "Select a pre-configured sample...": {
        "subject": "",
        "sender": "",
        "body": ""
    },
    "🚨 Phishing with 5 Embedded Links (Invoice Fraud)": {
        "subject": "URGENT: Outstanding Invoice Overdue - Immediate Payment Required",
        "sender": "accounts-billing@secure-corporate-notifications.xyz",
        "body": """Dear Customer,

Your corporate subscription account has been temporarily suspended due to 2 failed billing attempts for Invoice #98412 ($2,450.00).

To prevent permanent account termination and legal escalation within 24 hours, you must verify your identity and settle the invoice immediately:

1. Review detailed invoice breakdown: http://192.168.1.105/billing/invoice-98412.pdf
2. Authenticate your billing profile: https://secure-login.paypal.com.account-update.xyz/signin
3. Update corporate credit card: http://billing-portal-update.buzz/auth?token=92841
4. Contact verified support desk: https://www.google.com/search?q=corporate+billing
5. Read payment policy: https://en.wikipedia.org/wiki/Invoice

If you do not confirm within 24 hours, your access will be permanently locked.

Regards,
Corporate Billing Dept."""
    },
    "📧 Promotional Spam (Cash Prize / Lottery)": {
        "subject": "CONGRATULATIONS! You won $1,000,000 in International Cash Sweepstakes!",
        "sender": "rewards@lucky-winner-draw2026.net",
        "body": """CONGRATULATIONS!

You have been selected as our lucky first prize winner of $1,000,000 USD in our yearly cash lottery sweepstakes!
This exclusive payout is guaranteed with no risk.

To claim your prize, send your full bank routing numbers, passport copy and full address.
Act now! This limited time offer expires in 48 hours.

Click here to register your prize: http://claim-cash-bonus.party/winner"""
    },
    "✅ Legitimate Team Update (Safe with Benign Links)": {
        "subject": "Sprint Retrospective Notes and Next Iteration Goals",
        "sender": "sarah.jenkins@company-tech.org",
        "body": """Hi Team,

Great job on concluding Sprint 42! We successfully closed 18 feature tickets and resolved our open staging bugs.

Please review the retro meeting notes here: https://github.com/torvalds/linux
You can check out the new design system guidelines on our portal: https://docs.python.org/3/library/urllib.parse.html

Let's do our planning session tomorrow at 10:00 AM.

Best regards,
Sarah Jenkins
Lead Engineering Manager"""
    },
    "⚠️ Subtle Credential Phishing (Clean Text + Obfuscated Link)": {
        "subject": "Please review the updated employee handbook",
        "sender": "human-resources@mycompany-internal.com",
        "body": """Good morning,

We have updated the internal HR safety guidelines and vacation policy for the upcoming quarter.
All team members are requested to review the document and electronically sign:

Access Document: http://user:pass@employee-handbook-docusign.top/portal/auth

Thank you for your cooperation,
Human Resources"""
    }
}

# Sidebar Navigation & Stats
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("PhishGuard AI")
    st.caption("Multi-Vector Email & URL Threat Analyzer")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation",
        ["🛡️ Email Threat Scanner", "🔗 Direct URL Scanner", "⚖️ Model Training & Comparison", "📊 SQLite History & Logs", "⚙️ Tech Stack & Specs"]
    )
    
    st.markdown("---")
    st.subheader("System Telemetry")
    stats = get_system_stats()
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Total Scans", stats["total_scans"])
    col_s2.metric("Phishing Alerts", stats["phishing_count"])
    col_s3, col_s4 = st.columns(2)
    col_s3.metric("Spam Caught", stats["spam_count"])
    col_s4.metric("Malicious URLs", stats["malicious_urls"])

# ---------------------------------------------------------
# VIEW 1: EMAIL THREAT SCANNER
# ---------------------------------------------------------
if nav_option == "🛡️ Email Threat Scanner":
    st.title("🛡️ Unified Email Threat & Phishing Detection")
    st.markdown(
        "Analyzes raw emails across three vectors: **Text Spam (Naive Bayes)**, "
        "**Credential Phishing (Logistic Regression)**, and **Embedded Link Features (ML & Heuristics)**."
    )

    # Sample Selector
    selected_sample = st.selectbox("Quick-Load Sample Email Scenario:", list(SAMPLE_EMAILS.keys()))
    default_subj = SAMPLE_EMAILS[selected_sample]["subject"]
    default_sender = SAMPLE_EMAILS[selected_sample]["sender"]
    default_body = SAMPLE_EMAILS[selected_sample]["body"]

    with st.form("scan_form"):
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            input_sender = st.text_input("Sender Email / Header:", value=default_sender, placeholder="e.g. security-alert@paypal-update.com")
        with col_meta2:
            input_subject = st.text_input("Subject Line:", value=default_subj, placeholder="e.g. Urgent: Account Verification Required")
            
        input_body = st.text_area(
            "Email Body (Plaintext or HTML):",
            value=default_body,
            height=200,
            placeholder="Paste raw email content or HTML message here..."
        )
        
        submitted = st.form_submit_button("🔍 Run Multi-Vector Threat Analysis", use_container_width=True)

    if submitted:
        if not input_body.strip():
            st.warning("Please provide email body text to analyze.")
        else:
            with st.spinner("Analyzing message vectors and extracting embedded URLs..."):
                # 1. Feature 1: Embedded URL Extraction
                found_urls = extract_urls(input_body)
                
                # 2. Text Classification
                full_text_to_eval = f"{input_subject} {input_body}"
                spam_result = spam_detector.predict_spam(full_text_to_eval)
                phish_result = phishing_detector.predict_phishing(full_text_to_eval)
                
                # 3. Explainability on Text
                spam_exp = spam_detector.explain(full_text_to_eval)
                phish_exp = phishing_detector.explain(full_text_to_eval)
                
                # 4. Embedded URL Threat Analysis
                url_eval_results = []
                for u in found_urls:
                    u_res = url_detector.predict_url(u)
                    url_eval_results.append(u_res)
                    
                # 5. Feature 2: Combined Threat Report Synthesis
                report = synthesize_threat_report(
                    subject=input_subject,
                    sender=input_sender,
                    body=input_body,
                    spam_res=spam_result,
                    phish_res=phish_result,
                    url_results=url_eval_results,
                    spam_exp=spam_exp,
                    phish_exp=phish_exp
                )
                
                # 6. Storage in SQLite
                scan_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": input_subject,
                    "sender": input_sender,
                    "body_snippet": input_body[:250],
                    "spam_score": report["scores"]["spam_score"],
                    "phishing_score": report["scores"]["phishing_score"],
                    "url_max_score": report["scores"]["url_max_score"],
                    "composite_score": report["scores"]["composite_score"],
                    "verdict": report["verdict"],
                    "threat_level": report["threat_level"],
                    "summary": report
                }
                save_scan(scan_record, url_eval_results)

            st.success("Analysis Complete! Threat assessment rendered below.")
            
            # --- EXECUTIVE SUMMARY CARD ---
            st.markdown("---")
            st.subheader("🎯 Combined Threat Report")
            
            # Color-coded Banner based on threat level
            threat_color_map = {
                "CRITICAL": "#ff4b4b",
                "HIGH": "#ff7846",
                "MEDIUM": "#ffa500",
                "LOW": "#28a745"
            }
            border_color = threat_color_map.get(report["threat_level"], "#333")
            
            st.markdown(
                f"""
                <div style="padding: 18px; border-radius: 8px; border-left: 8px solid {border_color}; background-color: rgba(255, 255, 255, 0.05); margin-bottom: 20px;">
                    <div style="font-size: 22px; font-weight: bold; color: {border_color};">
                        {report["verdict"]}
                    </div>
                    <div style="font-size: 15px; margin-top: 8px;">
                        <strong>Action Guidance:</strong> {report["action_advice"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Metric Score Columns
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Composite Risk Score", f"{report['scores']['composite_percent']}%", delta=report["threat_level"], delta_color="inverse")
            m2.metric("Phishing Probability", f"{round(report['scores']['phishing_score'] * 100, 1)}%", help="TF-IDF + Logistic Regression")
            m3.metric("Spam Probability", f"{round(report['scores']['spam_score'] * 100, 1)}%", help="TF-IDF + Naive Bayes")
            m4.metric("Max URL Threat", f"{round(report['scores']['url_max_score'] * 100, 1)}%", f"{report['num_urls']} Links Found", delta_color="inverse")

            # --- STANDOUT FEATURE 1: EMBEDDED URL EXTRACTION & ANALYSIS ---
            st.markdown("---")
            st.subheader(f"🔗 Embedded URL Extraction ({report['num_urls']} Links Analyzed)")
            
            if report["num_urls"] == 0:
                st.info("No embedded web links detected in this message body.")
            else:
                st.write("All embedded links in the message were automatically parsed, defanged, and evaluated against lexical and structural indicators:")
                
                url_table_data = []
                for idx, u in enumerate(url_eval_results, start=1):
                    url_table_data.append({
                        "#": idx,
                        "URL": u["url"],
                        "Risk Score": f"{round(u['risk_score'] * 100, 1)}%",
                        "Status": "🚨 SUSPICIOUS" if u["is_suspicious"] else "✅ CLEAN",
                        "Red Flags Count": len(u["reasons"])
                    })
                st.dataframe(pd.DataFrame(url_table_data), use_container_width=True)
                
                # Detailed Expander per link
                with st.expander("🔍 Inspect Deep Structural Flags for Each Embedded Link"):
                    for idx, u in enumerate(url_eval_results, start=1):
                        st.markdown(f"**Link #{idx}:** `{u['url']}`")
                        if u["reasons"]:
                            for r in u["reasons"]:
                                st.markdown(f"- ⚠️ **Flag:** {r}")
                        else:
                            st.markdown("- ✅ *Passed all standard lexical checks (clean domain structure, standard port, no obfuscations).*")
                        st.markdown("---")

            # --- STANDOUT FEATURE 3: EXPLAINABLE AI (XAI) PANEL ---
            st.markdown("---")
            st.subheader("💡 Explainable AI (XAI): Threat Drivers & Feature Contributions")
            st.write("Understand the specific linguistic triggers and heuristic signals that influenced each model's prediction:")

            tab_xai1, tab_xai2, tab_xai3 = st.tabs([
                "🎣 Phishing Linguistic Drivers (Logistic Regression)",
                "📧 Spam Triggers (Naive Bayes)",
                "🌐 URL Lexical & Heuristic Explanations"
            ])

            with tab_xai1:
                st.markdown(f"**Executive Phishing Assessment:** {report['explainability']['text_analysis']['phish_summary']}")
                col_pt1, col_pt2 = st.columns(2)
                
                with col_pt1:
                    st.markdown("##### 🚨 Top Phishing Contributors (Positive Weights)")
                    pt = report["explainability"]["text_analysis"]["phish_triggers"]
                    if pt:
                        df_pt = pd.DataFrame(pt)
                        df_pt.columns = ["Token / N-Gram", "Impact Weight (w_i * tfidf)"]
                        st.table(df_pt)
                    else:
                        st.caption("No significant phishing n-grams detected.")

                with col_pt2:
                    st.markdown("##### 🛡️ Top Legitimate Indicators (Negative Weights)")
                    si = report["explainability"]["text_analysis"]["safe_indicators"]
                    if si:
                        df_si = pd.DataFrame(si)
                        df_si.columns = ["Token / N-Gram", "Impact Weight"]
                        st.table(df_si)
                    else:
                        st.caption("No strong legitimate tokens countered the classification.")

            with tab_xai2:
                st.markdown(f"**Executive Spam Assessment:** {report['explainability']['text_analysis']['spam_summary']}")
                col_st1, col_st2 = st.columns(2)
                
                with col_st1:
                    st.markdown("##### 📢 Promotional / Spam Triggers (Log-Odds Ratio)")
                    st_trig = report["explainability"]["text_analysis"]["spam_triggers"]
                    if st_trig:
                        df_st = pd.DataFrame(st_trig)
                        df_st.columns = ["Spam Token", "Log-Odds Difference"]
                        st.table(df_st)
                    else:
                        st.caption("No active spam vocabulary triggers found.")
                        
                with col_st2:
                    st.markdown("##### 💬 Normal / Ham Vocabulary Tokens")
                    ham_ind = report["explainability"]["text_analysis"]["ham_indicators"]
                    if ham_ind:
                        df_ham = pd.DataFrame(ham_ind)
                        df_ham.columns = ["Ham Token", "Relative Association"]
                        st.table(df_ham)
                    else:
                        st.caption("No strong ham indicators found.")

            with tab_xai3:
                st.markdown("##### 🌐 Structural & Lexical URL Risk Analysis")
                u_break = report["explainability"]["url_analysis"]["url_breakdown"]
                if not u_break:
                    st.info("No URLs to explain.")
                else:
                    for item in u_break:
                        st.markdown(f"**URL:** `{item['url']}` (Risk: `{round(item['risk_score']*100, 1)}%`)")
                        for reason in item["reasons"]:
                            st.write(f"- {reason}")

# ---------------------------------------------------------
# VIEW 2: DIRECT URL SCANNER
# ---------------------------------------------------------
elif nav_option == "🔗 Direct URL Scanner":
    st.title("🔗 Dedicated URL Feature & Heuristic Analyzer")
    st.write("Scan standalone URLs to extract 20+ lexical, structural, and behavioral threat features in real-time.")

    target_url = st.text_input(
        "Enter Target URL:",
        value="http://192.168.1.100/paypal-security/login.php?cmd=verify",
        placeholder="https://example.com/login"
    )

    if st.button("Analyze URL Features", use_container_width=True):
        if not target_url.strip():
            st.warning("Please provide a URL.")
        else:
            with st.spinner("Analyzing URL features..."):
                u_res = url_detector.predict_url(target_url.strip())
                f_dict = extract_url_feature_dict(target_url.strip())
                
            st.markdown("---")
            u_col1, u_col2 = st.columns([1, 2])
            with u_col1:
                st.metric("Malicious Risk Probability", f"{round(u_res['risk_score'] * 100, 1)}%")
                if u_res["is_suspicious"]:
                    st.error("🚨 VERDICT: SUSPICIOUS / MALICIOUS URL")
                else:
                    st.success("✅ VERDICT: LOW RISK / BENIGN URL")
                    
            with u_col2:
                st.subheader("🚩 Detected Anomaly Warnings")
                if u_res["reasons"]:
                    for r in u_res["reasons"]:
                        st.warning(f"⚠️ {r}")
                else:
                    st.info("No suspicious lexical patterns or structural red-flags detected.")

            st.markdown("---")
            st.subheader("📊 Extracted 20 Lexical & Structural Features")
            df_feat = pd.DataFrame(list(f_dict.items()), columns=["Feature Name", "Extracted Value"])
            st.dataframe(df_feat, use_container_width=True)

# ---------------------------------------------------------
# VIEW: MODEL TRAINING & BENCHMARKING
# ---------------------------------------------------------
elif nav_option == "⚖️ Model Training & Comparison":
    st.title("⚖️ Model Training & Algorithm Benchmarking")
    st.markdown(
        "Train, evaluate, and compare **Multinomial Naive Bayes** and **Logistic Regression** "
        "using **TF-IDF Vectorization** on labeled ground-truth datasets. Evaluate on Accuracy, Precision, Recall, "
        "F1-score, and Confusion Matrices."
    )

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        task_choice = st.selectbox(
            "Select Evaluation Task & Labeled Dataset:",
            ["Spam Detection (Spam vs Ham)", "Email Phishing Detection (Phishing vs Legitimate)"]
        )
    with col_t2:
        test_split = st.slider("Test Set Split Ratio:", min_value=0.15, max_value=0.40, value=0.25, step=0.05)

    dataset_map = {
        "Spam Detection (Spam vs Ham)": config.BASE_DIR / "data" / "spam_dataset.csv",
        "Email Phishing Detection (Phishing vs Legitimate)": config.BASE_DIR / "data" / "phishing_dataset.csv"
    }
    target_csv = dataset_map[task_choice]

    if st.button("🚀 Train & Compare Models", use_container_width=True):
        with st.spinner("Vectorizing text with TF-IDF, training classifiers, and computing confusion matrices..."):
            results = train_and_compare_models(
                csv_path=target_csv,
                task_name=task_choice,
                test_size=test_split
            )
            st.session_state["benchmark_results"] = results

    if "benchmark_results" in st.session_state:
        res = st.session_state["benchmark_results"]
        st.success(f"Benchmark completed on **{res['task_name']}**! (Total: {res['dataset_size']} samples, Train: {res['train_size']}, Test: {res['test_size']})")

        # 1. METRIC COMPARISON TABLE
        st.subheader("📋 Evaluation Metrics Comparison")
        st.dataframe(res["comparison_table"], use_container_width=True)

        # 2. SIDE BY SIDE KPI CARDS
        st.markdown("---")
        st.subheader("🎯 Side-by-Side Performance Comparison")
        col_nb, col_lr = st.columns(2)
        
        with col_nb:
            st.markdown("#### 🔵 Multinomial Naive Bayes (TF-IDF)")
            k1, k2 = st.columns(2)
            k1.metric("Accuracy", f"{res['naive_bayes']['accuracy'] * 100:.1f}%")
            k2.metric("F1-Score", f"{res['naive_bayes']['f1_score'] * 100:.1f}%")
            k3, k4 = st.columns(2)
            k3.metric("Precision", f"{res['naive_bayes']['precision'] * 100:.1f}%")
            k4.metric("Recall", f"{res['naive_bayes']['recall'] * 100:.1f}%")

        with col_lr:
            st.markdown("#### 🟢 Logistic Regression (TF-IDF)")
            l1, l2 = st.columns(2)
            l1.metric("Accuracy", f"{res['logistic_regression']['accuracy'] * 100:.1f}%")
            l2.metric("F1-Score", f"{res['logistic_regression']['f1_score'] * 100:.1f}%")
            l3, l4 = st.columns(2)
            l3.metric("Precision", f"{res['logistic_regression']['precision'] * 100:.1f}%")
            l4.metric("Recall", f"{res['logistic_regression']['recall'] * 100:.1f}%")

        # 3. CONFUSION MATRIX VISUALIZATION
        st.markdown("---")
        st.subheader("📊 Confusion Matrix Visualization")
        st.write("Heatmap of True Negatives, False Positives, False Negatives, and True Positives:")
        if Path(res["confusion_matrix_image"]).exists():
            st.image(res["confusion_matrix_image"], use_column_width=True)

        # 4. LIVE INFERENCE COMPARISON TOOL
        st.markdown("---")
        st.subheader("🧪 Live Comparative Inference Test")
        st.write("Test any custom message text on **both** models simultaneously to compare their probabilistic outputs:")
        
        test_input = st.text_input(
            "Enter sentence to test:",
            value="Urgent: Your account has been restricted. Confirm your login password within 24 hours.",
            key="bench_test_input"
        )
        if test_input:
            vec = res["vectorizer"]
            feat = vec.transform([test_input])
            nb_prob = res["nb_model"].predict_proba(feat)[0][1]
            lr_prob = res["lr_model"].predict_proba(feat)[0][1]

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.metric("Naive Bayes Threat Probability", f"{nb_prob * 100:.1f}%")
                if nb_prob >= 0.5:
                    st.error("🚨 Naive Bayes: Malicious / Positive")
                else:
                    st.success("✅ Naive Bayes: Safe / Negative")
            with t_col2:
                st.metric("Logistic Regression Threat Probability", f"{lr_prob * 100:.1f}%")
                if lr_prob >= 0.5:
                    st.error("🚨 Logistic Regression: Malicious / Positive")
                else:
                    st.success("✅ Logistic Regression: Safe / Negative")

# ---------------------------------------------------------
# VIEW 3: SQLITE HISTORY & LOGS
# ---------------------------------------------------------
elif nav_option == "📊 SQLite History & Logs":
    st.title("📊 Scan History & Security Logs (SQLite)")
    st.write(f"Persistent audit database located at: `{config.DATABASE_PATH}`")
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("🗑️ Clear Scan History"):
            clear_all_history()
            st.success("Database history wiped!")
            st.rerun()

    scans = get_recent_scans(limit=100)
    if not scans:
        st.info("No scans recorded yet. Run a scan from the 'Email Threat Scanner' tab.")
    else:
        df_scans = pd.DataFrame(scans)
        st.dataframe(df_scans, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Inspect Detailed Scan Record")
        scan_ids = [s["id"] for s in scans]
        chosen_id = st.selectbox("Select Scan ID to Inspect:", scan_ids)
        
        details = get_scan_details(chosen_id)
        if details:
            st.json({
                "ID": details["id"],
                "Timestamp": details["timestamp"],
                "Subject": details["subject"],
                "Sender": details["sender"],
                "Verdict": details["verdict"],
                "Threat Level": details["threat_level"],
                "Scores": {
                    "Composite": details["composite_score"],
                    "Phishing": details["phishing_score"],
                    "Spam": details["spam_score"],
                    "Max URL": details["url_max_score"]
                },
                "Embedded URLs Found": details["urls"]
            })

# ---------------------------------------------------------
# VIEW 4: TECH STACK & SYSTEM ARCHITECTURE
# ---------------------------------------------------------
elif nav_option == "⚙️ Tech Stack & Specs":
    st.title("⚙️ System Architecture & Technology Specifications")
    
    st.markdown("""
    ### 🧱 Technical Foundation
    | Component | Technology / Implementation |
    | :--- | :--- |
    | **Frontend** | Streamlit (Interactive Multi-tab Interface) |
    | **Backend** | Python 3.11+ |
    | **Spam Detection** | TF-IDF Vectorizer + Multinomial Naive Bayes |
    | **Email Phishing** | TF-IDF Vectorizer + Logistic Regression |
    | **URL Threat Detection** | 20+ Structural & Lexical Feature Extraction + Random Forest ML |
    | **Data Processing** | Pandas & NumPy |
    | **Evaluation & Metrics** | Scikit-learn (`classification_report`, `accuracy`, `precision_recall_fscore`) |
    | **Explainable AI (XAI)** | MultinomialNB log-odds, LogisticRegression linear weights & URL heuristic engine |
    | **Data Persistence** | SQLite (`phish_guard.db`) with relational scan and URL history |
    
    ---
    
    ### 🌟 Core Differentiators & Standout Features
    
    #### 1. Embedded URL Extraction
    - Parses both raw plaintext links, href attributes, and defanged URLs (`hxxp://`, `[.]`, etc.).
    - Extracts all links even if an email embeds 5, 10, or more links.
    - Evaluates each URL individually, extracting domain structures, subdomains, TLDs, and path entropy.
    
    #### 2. Combined Multi-Vector Threat Report
    - A polite message body with a weaponized link is a phishing attack!
    - Threat Aggregator synthesizes Text Spam ($P_{spam}$), Phishing Intent ($P_{phish}$), and Max URL Threat ($P_{url\_max}$).
    - Enforces critical safety overrides so weaponized URLs escalate the entire message to **CRITICAL**.
    
    #### 3. Explainable AI (XAI)
    - Demystifies the "black box":
      - Pinpoints exact tokens driving the spam and phishing flags.
      - Displays human-readable warnings for URL anomalies (e.g. raw IP address, typo-squatting hyphens, suspicious `.xyz` TLD).
    """)

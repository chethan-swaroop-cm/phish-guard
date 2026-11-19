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
    init_db, save_scan, get_recent_scans, get_scan_details, get_system_stats, clear_all_history,
    save_fraud_report, get_fraud_reports
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
    page_title="PhishGuard AI — Cyber Threat Radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# AESTHETIC CYBER DASHBOARD STYLING & ANIMATIONS (CSS)
# ---------------------------------------------------------
AESTHETIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Master Theme Variables */
:root {
    --bg-main: #060913;
    --bg-card: #0d1424;
    --bg-card-hover: #131c31;
    --border-color: rgba(0, 242, 254, 0.2);
    --border-hover: rgba(0, 242, 254, 0.45);
    --cyan: #00f2fe;
    --blue: #3b82f6;
    --crimson: #ff0055;
    --amber: #f59e0b;
    --emerald: #10b981;
    --purple: #8b5cf6;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

/* Global Background and Typography */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    background: radial-gradient(circle at 15% 15%, #0f1c38 0%, #060913 60%, #03050a 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main) !important;
}

/* Hide Streamlit Header Bar Artifacts */
header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: blur(8px);
}
#MainMenu, footer {visibility: hidden;}

/* Sleek Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 29, 0.96) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.12) !important;
    backdrop-filter: blur(20px);
}

/* Fix Streamlit Input Colors (No more stark white boxes!) */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stSelectbox>div>div {
    background: #0d1527 !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    transition: all 0.25s ease !important;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.3) !important;
}

/* Fix Labels */
label, .stSelectbox label, .stTextInput label, .stTextArea label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
}

/* Radio Navigation Styling */
.stRadio [role="radiogroup"] {
    gap: 8px;
}
.stRadio label {
    background: rgba(13, 21, 39, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    display: flex;
    align-items: center;
    color: #cbd5e1 !important;
    font-size: 13px !important;
}
.stRadio label:hover {
    border-color: rgba(0, 242, 254, 0.4) !important;
    background: rgba(0, 242, 254, 0.08) !important;
    color: #ffffff !important;
}

/* Action Button */
.stButton>button {
    background: linear-gradient(135deg, #00f2fe 0%, #3b82f6 100%) !important;
    color: #020617 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 28px !important;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton>button:hover {
    box-shadow: 0 6px 30px rgba(0, 242, 254, 0.6) !important;
    transform: translateY(-2px) !important;
    color: #000000 !important;
}

/* Hero Section with Animation */
.hero-box {
    background: linear-gradient(135deg, rgba(13, 22, 45, 0.85) 0%, rgba(9, 14, 28, 0.95) 100%);
    border: 1px solid rgba(0, 242, 254, 0.22);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 0 30px rgba(0, 242, 254, 0.04);
}
.hero-box::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(0, 242, 254, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
    animation: pulseGlow 6s ease-in-out infinite alternate;
}
@keyframes pulseGlow {
    0% { transform: scale(0.9); opacity: 0.6; }
    100% { transform: scale(1.2); opacity: 0.9; }
}

/* Floating animation for badge icons */
@keyframes floatShield {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-6px) rotate(1deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}
.floating-icon {
    animation: floatShield 4s ease-in-out infinite;
    display: inline-block;
}

/* Feature Showcase Cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.feature-card {
    background: rgba(13, 20, 36, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: rgba(0, 242, 254, 0.4);
    background: rgba(13, 20, 36, 0.95);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 242, 254, 0.12);
}
.feature-title {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.feature-desc {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.4;
}

/* Metric KPI Cards */
.kpi-card {
    background: rgba(13, 21, 39, 0.85);
    border: 1px solid rgba(0, 242, 254, 0.16);
    border-radius: 12px;
    padding: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    border-color: rgba(0, 242, 254, 0.4);
    transform: translateY(-2px);
}
.kpi-title {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    margin: 6px 0 2px 0;
}
.kpi-sub {
    font-size: 11px;
    color: #64748b;
}

/* Threat Alert Banners */
.alert-banner-critical {
    background: linear-gradient(135deg, rgba(255, 0, 85, 0.22) 0%, rgba(30, 10, 20, 0.8) 100%);
    border: 1px solid #ff0055;
    border-radius: 12px;
    padding: 22px;
    box-shadow: 0 0 35px rgba(255, 0, 85, 0.25);
    margin-bottom: 24px;
}
.alert-banner-clean {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(10, 25, 20, 0.8) 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 22px;
    box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
    margin-bottom: 24px;
}
.alert-banner-amber {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(25, 20, 10, 0.8) 100%);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 22px;
    box-shadow: 0 0 35px rgba(245, 158, 11, 0.2);
    margin-bottom: 24px;
}

/* Pulsing Online Radar Dot */
.pulsing-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: dotPulse 1.5s infinite;
}
@keyframes dotPulse {
    0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
</style>
"""
st.markdown(AESTHETIC_CSS, unsafe_allow_html=True)

# Initialize DB and Models
@st.cache_resource
def load_models():
    init_db()
    if not (config.SPAM_MODEL_PATH.exists() and config.PHISHING_MODEL_PATH.exists() and config.URL_MODEL_PATH.exists()):
        train_and_save_all_models()
        
    spam_det = SpamDetector().load(config.SPAM_MODEL_PATH)
    phish_det = PhishingDetector().load(config.PHISHING_MODEL_PATH)
    url_det = URLDetector().load(config.URL_MODEL_PATH)
    return spam_det, phish_det, url_det

try:
    spam_detector, phishing_detector, url_detector = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ---------------------------------------------------------
# EMERGENCY CYBER CRIME MODAL DIALOG (HELPLINE 1930)
# ---------------------------------------------------------
@st.dialog("🚨 EMERGENCY CYBER CRIME ESCALATION // HELPLINE 1930")
def show_cyber_crime_dialog(report_summary: dict, sender: str, subject: str, body: str, urls: list):
    st.markdown("""
        <div style="background: rgba(255, 0, 85, 0.12); border-left: 4px solid #ff0055; padding: 14px; border-radius: 8px; margin-bottom: 16px;">
            <div style="font-size: 16px; font-weight: 800; color: #ff0055;">⚠️ HIGH-SEVERITY PHISHING / FRAUD DETECTED (> 70%)</div>
            <div style="font-size: 13px; color: #e2e8f0; margin-top: 4px;">
                This communication has been assessed with over <strong>70% threat probability</strong>. You can report the suspicious sender and spam message directly to the National Cyber Crime Bureau.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📞 1. Instant Phone App Escalation (Helpline 1930)")
    st.write("Click below to open your phone app/dialer and immediately call the **National Cyber Crime Helpline (1930)**:")
    st.markdown("""
        <a href="tel:1930" style="
            display: block;
            text-align: center;
            background: linear-gradient(90deg, #ff0055 0%, #ff4b4b 100%);
            color: #ffffff;
            font-weight: 800;
            font-size: 16px;
            padding: 14px 20px;
            border-radius: 8px;
            text-decoration: none;
            box-shadow: 0 4px 20px rgba(255, 0, 85, 0.4);
            margin-bottom: 16px;
        ">
            📞 CALL CYBER CRIME POLICE (1930) NOW
        </a>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🏛️ 2. Official National Cyber Crime Reporting Portal")
    st.markdown("""
        <a href="https://cybercrime.gov.in/" target="_blank" style="
            display: block;
            text-align: center;
            background: rgba(0, 242, 254, 0.12);
            border: 1px solid #00f2fe;
            color: #00f2fe;
            font-weight: 700;
            font-size: 14px;
            padding: 12px 20px;
            border-radius: 8px;
            text-decoration: none;
            margin-bottom: 20px;
        ">
            🌐 OPEN CYBERCRIME.GOV.IN (FILE ONLINE COMPLAINT)
        </a>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 3. Pre-formatted Evidence Dossier")
    st.write("Copy and paste this structured evidence when lodging your formal complaint:")
    
    url_formatted = "\\n  - ".join(urls) if urls else "None detected"
    evidence_text = f"""INCIDENT REPORT // CYBER FRAUD EVIDENCE
Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST
Reported Suspect Sender: {sender or 'Unknown Sender'}
Subject Line: {subject or 'No Subject'}
Assessed Threat Level: {report_summary.get('composite_percent', 0)}%
Verdict: {report_summary.get('verdict', 'CRITICAL PHISHING')}
Weaponized / Deceptive Links:
  - {url_formatted}

Extracted Email Payload:
{body[:500]}
"""
    st.text_area("Evidence Text Box:", value=evidence_text, height=160)
    
    if st.button("💾 Log Incident in Local Police Audit Records", use_container_width=True):
        docket_id = save_fraud_report({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender,
            "subject": subject,
            "message_snippet": body[:500],
            "phishing_score": report_summary.get("phishing_score", 0.0),
            "composite_score": report_summary.get("composite_score", 0.0),
            "urls": urls,
            "status": "ESCALATED_TO_1930_HELPLINE"
        })
        st.success(f"✅ Fraud incident successfully archived! Reference Docket ID: **{docket_id}**")

# Cyber-Themed Sample Scenarios
SAMPLE_EMAILS = {
    "Select an attack simulation scenario...": {
        "subject": "",
        "sender": "",
        "body": ""
    },
    "🚨 Weaponized Invoice Fraud (5 Embedded Links / IP Host)": {
        "subject": "CRITICAL: Corporate Invoice #INV-9942 Overdue - Immediate Account Suspension",
        "sender": "sec-billing@enterprise-notifications-update.xyz",
        "body": """CONFIDENTIAL SECURITY NOTICE // CORP BILLING

Your corporate SaaS enterprise subscription has been quarantined due to 2 failed credit settlements for Invoice #INV-9942 ($2,450.00).

To abort complete token revocation and legal compliance escalation within 24 hours, authenticate your credential profile and confirm transaction details immediately:

1. Review raw invoice audit PDF: http://192.168.1.105/billing/invoice-98412.pdf
2. Verify administrative corporate profile: https://secure-login.paypal.com.account-update.xyz/signin
3. Re-enter root credit card authentication token: http://billing-portal-update.buzz/auth?token=92841
4. Inspect corporate directory support: https://www.google.com/search?q=corporate+billing
5. Review digital transaction treaty: https://en.wikipedia.org/wiki/Invoice

Failure to confirm within 24 hours will trigger irreversible credential purge.

Operations Security Dept // Cyber Defense Division"""
    },
    "📢 Mass Bulk Spam (Cryptocurrency & Cash Prize Scheme)": {
        "subject": "CONGRATULATIONS! $1,000,000 USD International Crypto Lottery Granted!",
        "sender": "rewards@lucky-draw-bitcoin2026.net",
        "body": """CONGRATULATIONS WINNER!

You have been selected as the prime recipient of $1,000,000 USD in our automated cryptographic lottery draw!
This exclusive jackpot is guaranteed with zero risk and instantaneous blockchain payout.

To claim your allocation, transmit full bank wire routing numbers, identity passport, and mailing address.
ACT NOW! This time-gated promotional window expires in 48 hours.

Register claim code: http://claim-cash-bonus.party/winner"""
    },
    "🛡️ Verified Corporate Transmission (Benign + Safe Links)": {
        "subject": "Sprint 44 Engineering Review & Architecture Roadmap",
        "sender": "sarah.jenkins@company-tech.org",
        "body": """Hi Team,

Great execution on finalizing Sprint 44 milestones. We have closed 19 pull requests and optimized the ingestion worker throughput by 32%.

Please inspect the architecture diagrams here: https://github.com/torvalds/linux
You can also reference the revised parsing specification: https://docs.python.org/3/library/urllib.parse.html

Our strategy briefing will occur tomorrow at 10:00 AM UTC.

Best regards,
Sarah Jenkins
Lead Infrastructure Architect"""
    },
    "⚠️ Stealth Social Engineering (Clean Lure + Obfuscated Auth URL)": {
        "subject": "Mandatory Q4 Compensation Adjustment & Employee Handbook",
        "sender": "human-resources@mycompany-internal.com",
        "body": """Team Members,

Executive leadership has released the revised Q4 compensation matrices and health insurance policy updates.
Every employee is requested to review the documentation and acknowledge signature:

Review Document: http://user:pass@employee-handbook-docusign.top/portal/auth

Thank you for your timely attention,
Human Resources Division"""
    }
}

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 12px 0 16px 0;">
            <div class="floating-icon" style="font-size: 42px;">🛡️</div>
            <div style="font-size: 22px; font-weight: 800; letter-spacing: 1.5px; background: linear-gradient(90deg, #00f2fe, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 6px;">
                PHISHGUARD AI
            </div>
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #64748b; letter-spacing: 1px; margin-top: 2px;">
                CYBER THREAT RADAR v2.5
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); border-radius: 6px; padding: 6px 12px; margin-bottom: 20px;">
            <span class="pulsing-dot"></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #10b981; font-weight: 600;">ACTIVE DEFENSE // ONLINE</span>
        </div>
    """, unsafe_allow_html=True)

    nav_option = st.radio(
        "NAVIGATION COMMAND",
        [
            "🛡️ Email Threat Scanner",
            "🔗 Tactical URL Matrix",
            "⚖️ Model Benchmarking (ML)",
            "📊 SQLite Forensic Logs",
            "⚙️ Architecture Specs"
        ]
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 11px; font-family: \"JetBrains Mono\", monospace; color: #00f2fe; margin-bottom: 10px; font-weight: 700;'>SECURITY TELEMETRY</div>", unsafe_allow_html=True)
    stats = get_system_stats()
    c_s1, c_s2 = st.columns(2)
    c_s1.metric("Scans Logged", stats["total_scans"])
    c_s2.metric("Threats Flagged", stats["phishing_count"])
    c_s3, c_s4 = st.columns(2)
    c_s3.metric("Spam Trapped", stats["spam_count"])
    c_s4.metric("Malicious URLs", stats["malicious_urls"])

    st.markdown("---")
    st.caption("Engine: TF-IDF + Naive Bayes + Logistic Regression + Random Forest + XAI Attribution")

# ---------------------------------------------------------
# VIEW 1: EMAIL THREAT SCANNER
# ---------------------------------------------------------
if nav_option == "🛡️ Email Threat Scanner":
    # Hero Banner with Vector Art & Visuals
    st.markdown("""
        <div class="hero-box">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 28px;">🛡️</span>
                        <h1 style="font-size: 26px; font-weight: 800; margin: 0; color: #ffffff; letter-spacing: -0.5px;">
                            Autonomous Email Threat Radar
                        </h1>
                    </div>
                    <p style="color: #94a3b8; font-size: 14px; margin: 8px 0 0 0; max-width: 650px;">
                        Multi-vector defensive intelligence engine. Simultaneously extracts and investigates embedded links, evaluates credential solicitation language, and calculates unified threat risk.
                    </p>
                </div>
                <div style="background: rgba(0,242,254,0.06); border: 1px solid rgba(0,242,254,0.2); border-radius: 10px; padding: 12px 18px; text-align: center;">
                    <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #00f2fe; font-weight: 600;">ACTIVE DEFENSE MODE</div>
                    <div style="font-size: 18px; font-weight: 800; color: #ffffff; margin-top: 2px;">TRI-VECTOR XAI</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3 Standout Feature Cards
    st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-title"><span>🔗</span> 1. Multi-Link Extractor</div>
                <div class="feature-desc">Automatically parses every embedded link, defanging obfuscations and scoring 20+ lexical anomalies.</div>
            </div>
            <div class="feature-card">
                <div class="feature-title"><span>🎯</span> 2. Combined Threat Matrix</div>
                <div class="feature-desc">Combines text spam, phishing intent, and URL risks into a weighted composite score with safety overrides.</div>
            </div>
            <div class="feature-card">
                <div class="feature-title"><span>💡</span> 3. Explainable AI (XAI)</div>
                <div class="feature-desc">Breaks down exact token attributions and domain red flags (IP hosts, TLD abuse, subdomain nesting).</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Simulation preset selector
    selected_sample = st.selectbox(
        "⚡ Quick-Load Attack Simulation Preset:",
        list(SAMPLE_EMAILS.keys())
    )
    default_subj = SAMPLE_EMAILS[selected_sample]["subject"]
    default_sender = SAMPLE_EMAILS[selected_sample]["sender"]
    default_body = SAMPLE_EMAILS[selected_sample]["body"]

    with st.form("cyber_scan_form"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            input_sender = st.text_input("Origin Header / Sender Address:", value=default_sender, placeholder="e.g., billing@security-portal.xyz")
        with col_m2:
            input_subject = st.text_input("Subject Line:", value=default_subj, placeholder="e.g., URGENT: Credentials Suspended")
            
        input_body = st.text_area(
            "Email Payload (Plaintext or HTML):",
            value=default_body,
            height=200,
            placeholder="Paste email text or raw MIME content here..."
        )
        
        scan_btn = st.form_submit_button("⚡ EXECUTE MULTI-VECTOR THREAT AUDIT", use_container_width=True)

    if scan_btn:
        if not input_body.strip():
            st.warning("⚠️ Please provide email content to analyze.")
        else:
            with st.spinner("Executing tri-vector neural & heuristic scan..."):
                # 1. Feature 1: Embedded URLs
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
                    
                # 5. Feature 2: Threat Aggregator
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
                
                # 6. SQLite Logging
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

            st.markdown("---")
            threat_level = report["threat_level"]
            
            if threat_level == "CRITICAL":
                banner_class = "alert-banner-critical"
                icon = "🚨"
                accent_color = "#ff0055"
            elif threat_level in ("HIGH", "MEDIUM"):
                banner_class = "alert-banner-amber"
                icon = "⚠️"
                accent_color = "#f59e0b"
            else:
                banner_class = "alert-banner-clean"
                icon = "✅"
                accent_color = "#10b981"

            # Executive Threat Banner
            st.markdown(f"""
                <div class="{banner_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: {accent_color}; font-weight: 700; letter-spacing: 1.5px;">
                                // THREAT ASSESSMENT // VERDICT: {threat_level}
                            </div>
                            <div style="font-size: 22px; font-weight: 800; color: #ffffff; margin-top: 4px;">
                                {icon} {report["verdict"]}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8;">COMPOSITE RISK SCORE</div>
                            <div style="font-size: 34px; font-weight: 900; font-family: 'JetBrains Mono', monospace; color: {accent_color};">
                                {report['scores']['composite_percent']}%
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 14px; color: #e2e8f0;">
                        <strong>DEFENSIVE GUIDANCE:</strong> {report["action_advice"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Four High-Tech KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 3px solid {accent_color};">
                        <div class="kpi-title">Composite Risk</div>
                        <div class="kpi-value" style="color: {accent_color};">{report['scores']['composite_percent']}%</div>
                        <div class="kpi-sub">Unified Multi-Vector Hazard</div>
                    </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 3px solid #00f2fe;">
                        <div class="kpi-title">Phishing Intent</div>
                        <div class="kpi-value" style="color: #00f2fe;">{round(report['scores']['phishing_score'] * 100, 1)}%</div>
                        <div class="kpi-sub">TF-IDF + Logistic Regression</div>
                    </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 3px solid #f59e0b;">
                        <div class="kpi-title">Spam Index</div>
                        <div class="kpi-value" style="color: #f59e0b;">{round(report['scores']['spam_score'] * 100, 1)}%</div>
                        <div class="kpi-sub">TF-IDF + Naive Bayes</div>
                    </div>
                """, unsafe_allow_html=True)
            with k4:
                url_color = "#ff0055" if report['scores']['url_max_score'] >= 0.5 else "#10b981"
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 3px solid {url_color};">
                        <div class="kpi-title">Max URL Risk</div>
                        <div class="kpi-value" style="color: {url_color};">{round(report['scores']['url_max_score'] * 100, 1)}%</div>
                        <div class="kpi-sub">{report['num_urls']} Hyperlinks Analyzed</div>
                    </div>
                """, unsafe_allow_html=True)

            # --- EMERGENCY CYBER CRIME HELPLINE (1930) & FRAUD REPORTING TRIGGER (> 70%) ---
            is_high_risk = (report['scores']['composite_percent'] >= 70.0) or (report['scores']['phishing_score'] >= 0.70)
            if is_high_risk:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(255, 0, 85, 0.22) 0%, rgba(30, 10, 20, 0.95) 100%); border: 2px solid #ff0055; border-radius: 12px; padding: 18px 22px; box-shadow: 0 0 30px rgba(255, 0, 85, 0.35); margin-top: 20px; margin-bottom: 22px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                            <div>
                                <span style="background: #ff0055; color: #fff; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 4px; font-family: 'JetBrains Mono';">CRITICAL FRAUD THRESHOLD EXCEEDED (> 70%)</span>
                                <div style="font-size: 20px; font-weight: 800; color: #ffffff; margin-top: 6px;">
                                    🚨 Cyber Crime Escalation Protocol Triggered
                                </div>
                                <div style="color: #cbd5e1; font-size: 13px; margin-top: 4px;">
                                    This message exhibits aggressive social engineering or credential theft indicators. You can report this sender and dial the Cyber Police directly.
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 11px; color: #fecdd3; font-family: 'JetBrains Mono';">NATIONAL CYBER CRIME HELPLINE</div>
                                <div style="font-size: 28px; font-weight: 900; color: #ff0055; font-family: 'JetBrains Mono';">DIAL 1930</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                col_esc1, col_esc2 = st.columns(2)
                with col_esc1:
                    st.markdown("""
                        <a href="tel:1930" style="display: block; text-align: center; background: linear-gradient(90deg, #ff0055 0%, #ff4b4b 100%); color: #ffffff; font-weight: 800; font-size: 15px; padding: 12px 18px; border-radius: 8px; text-decoration: none; box-shadow: 0 4px 15px rgba(255, 0, 85, 0.4);">
                            📞 CALL CYBER CRIME POLICE (1930) NOW
                        </a>
                    """, unsafe_allow_html=True)
                with col_esc2:
                    if st.button("🚨 REPORT SENDER & OPEN COMPLAINT DOSSIER", use_container_width=True):
                        show_cyber_crime_dialog(
                            report_summary=report['scores'],
                            sender=input_sender,
                            subject=input_subject,
                            body=input_body,
                            urls=[u['url'] for u in url_eval_results]
                        )

            # Feature 1: Embedded URLs
            st.markdown("---")
            st.markdown(f"""
                <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
                    🔗 Embedded Hyperlink Forensic Dossier ({report['num_urls']} Links Detected)
                </div>
            """, unsafe_allow_html=True)

            if report["num_urls"] == 0:
                st.info("No embedded URLs were located in this message.")
            else:
                for idx, u in enumerate(url_eval_results, start=1):
                    risk_pct = round(u['risk_score'] * 100, 1)
                    is_susp = u['is_suspicious']
                    badge = "🚨 MALICIOUS" if is_susp else "✅ CLEAN"
                    
                    with st.expander(f"Link #{idx} // [{risk_pct}% Risk] {u['url']}", expanded=is_susp):
                        u1, u2 = st.columns([1, 2])
                        with u1:
                            st.write(f"**Status:** `{badge}`")
                            st.write(f"**Risk Score:** `{risk_pct}%`")
                            st.write(f"**URL:** `{u['url']}`")
                        with u2:
                            st.write("**Detected Structural & Lexical Anomalies:**")
                            if u["reasons"]:
                                for r in u["reasons"]:
                                    st.write(f"- ⚠️ **Flag:** {r}")
                            else:
                                st.write("- ✅ *Passed all standard lexical checks (clean domain, standard port, normal entropy).*")

            # Feature 3: Explainable AI (XAI)
            st.markdown("---")
            st.markdown("""
                <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
                    💡 Explainable AI (XAI): Linguistic & Structural Evidence
                </div>
            """, unsafe_allow_html=True)

            xai1, xai2, xai3 = st.tabs([
                "🎣 Phishing Evidence (Logistic Regression)",
                "📧 Spam Triggers (Naive Bayes)",
                "🌐 URL Anomaly Breakdown"
            ])

            with xai1:
                st.caption(f"Reasoning: {report['explainability']['text_analysis']['phish_summary']}")
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("##### 🚨 Top Phishing Contributors (Positive Weights)")
                    pt = report["explainability"]["text_analysis"]["phish_triggers"]
                    if pt:
                        for item in pt:
                            st.markdown(f"""
                                <div style="display:flex; justify-content:space-between; background:rgba(255,0,85,0.08); border-left:3px solid #ff0055; padding:6px 12px; margin-bottom:6px; border-radius:4px;">
                                    <span style="font-family:'JetBrains Mono'; color:#fff;">{item['token']}</span>
                                    <span style="font-family:'JetBrains Mono'; color:#ff0055; font-weight:bold;">+{item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No significant phishing n-grams detected.")
                with p2:
                    st.markdown("##### 🛡️ Top Legitimate Indicators (Negative Weights)")
                    si = report["explainability"]["text_analysis"]["safe_indicators"]
                    if si:
                        for item in si:
                            st.markdown(f"""
                                <div style="display:flex; justify-content:space-between; background:rgba(16,185,129,0.08); border-left:3px solid #10b981; padding:6px 12px; margin-bottom:6px; border-radius:4px;">
                                    <span style="font-family:'JetBrains Mono'; color:#fff;">{item['token']}</span>
                                    <span style="font-family:'JetBrains Mono'; color:#10b981; font-weight:bold;">-{item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No strong legitimate tokens countered the classification.")

            with xai2:
                st.caption(f"Reasoning: {report['explainability']['text_analysis']['spam_summary']}")
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown("##### 📢 Promotional Spam Vocabulary (Log-Odds)")
                    st_trig = report["explainability"]["text_analysis"]["spam_triggers"]
                    if st_trig:
                        for item in st_trig:
                            st.markdown(f"""
                                <div style="display:flex; justify-content:space-between; background:rgba(245,158,11,0.08); border-left:3px solid #f59e0b; padding:6px 12px; margin-bottom:6px; border-radius:4px;">
                                    <span style="font-family:'JetBrains Mono'; color:#fff;">{item['token']}</span>
                                    <span style="font-family:'JetBrains Mono'; color:#f59e0b; font-weight:bold;">Δ {item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No active promotional vocabulary triggers found.")
                with s2:
                    st.markdown("##### 💬 Normal Ham Vocabulary")
                    ham_ind = report["explainability"]["text_analysis"]["ham_indicators"]
                    if ham_ind:
                        for item in ham_ind:
                            st.markdown(f"""
                                <div style="display:flex; justify-content:space-between; background:rgba(0,242,254,0.08); border-left:3px solid #00f2fe; padding:6px 12px; margin-bottom:6px; border-radius:4px;">
                                    <span style="font-family:'JetBrains Mono'; color:#fff;">{item['token']}</span>
                                    <span style="font-family:'JetBrains Mono'; color:#00f2fe; font-weight:bold;">{item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No strong ham indicators found.")

            with xai3:
                u_break = report["explainability"]["url_analysis"]["url_breakdown"]
                if not u_break:
                    st.info("No URLs to explain.")
                else:
                    for item in u_break:
                        st.markdown(f"**Link:** `{item['url']}` (Risk: `{round(item['risk_score']*100, 1)}%`)")
                        for reason in item["reasons"]:
                            st.write(f"- {reason}")

# ---------------------------------------------------------
# VIEW 2: TACTICAL URL MATRIX
# ---------------------------------------------------------
elif nav_option == "🔗 Tactical URL Matrix":
    st.markdown("""
        <div class="hero-box">
            <h2 style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 0;">
                🔗 Standalone URL Forensic Matrix
            </h2>
            <p style="color: #94a3b8; font-size: 13px; margin: 6px 0 0 0;">
                Extracts 20+ lexical, structural, and Shannon entropy indicators from standalone URLs in real time.
            </p>
        </div>
    """, unsafe_allow_html=True)

    target_url = st.text_input(
        "Enter Target URL to Probe:",
        value="http://192.168.1.100/paypal-security/login.php?cmd=verify",
        placeholder="https://suspicious-domain.xyz/auth"
    )

    if st.button("PROBE URL MATRIX", use_container_width=True):
        if not target_url.strip():
            st.warning("Please supply a URL.")
        else:
            with st.spinner("Extracting structural features and scoring heuristic weights..."):
                u_res = url_detector.predict_url(target_url.strip())
                f_dict = extract_url_feature_dict(target_url.strip())
                
            risk_pct = round(u_res['risk_score'] * 100, 1)
            is_susp = u_res["is_suspicious"]
            
            c_u1, c_u2 = st.columns([1, 2])
            with c_u1:
                st.markdown(f"""
                    <div class="kpi-card" style="text-align: center; padding: 24px;">
                        <div class="kpi-title">MALICIOUS PROBABILITY</div>
                        <div style="font-size: 42px; font-weight: 900; font-family: 'JetBrains Mono', monospace; color: {'#ff0055' if is_susp else '#10b981'}; margin: 12px 0;">
                            {risk_pct}%
                        </div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 700; color: {'#ff0055' if is_susp else '#10b981'};">
                            {'🚨 VERDICT: MALICIOUS' if is_susp else '✅ VERDICT: BENIGN'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with c_u2:
                st.markdown("### 🚩 Detected Heuristic Anomalies")
                if u_res["reasons"]:
                    for r in u_res["reasons"]:
                        st.markdown(f"""
                            <div style="background: rgba(255,0,85,0.08); border-left: 3px solid #ff0055; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                                ⚠️ {r}
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ Clean URL. Passed all standard lexical checks.")

            st.markdown("---")
            st.subheader("📊 20 Granular Structural Metrics")
            df_feat = pd.DataFrame(list(f_dict.items()), columns=["Metric Key", "Computed Value"])
            st.dataframe(df_feat, use_container_width=True)

# ---------------------------------------------------------
# VIEW 3: MODEL BENCHMARKING (ML)
# ---------------------------------------------------------
elif nav_option == "⚖️ Model Benchmarking (ML)":
    st.markdown("""
        <div class="hero-box">
            <h2 style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 0;">
                ⚖️ Algorithm Benchmark & Comparison
            </h2>
            <p style="color: #94a3b8; font-size: 13px; margin: 6px 0 0 0;">
                TF-IDF Vectorization // Multinomial Naive Bayes vs. Logistic Regression // Confusion Matrices
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        task_choice = st.selectbox(
            "Select Benchmark Dataset:",
            ["Spam Detection (Spam vs Ham)", "Email Phishing Detection (Phishing vs Legitimate)"]
        )
    with col_t2:
        test_split = st.slider("Train/Test Split:", min_value=0.15, max_value=0.40, value=0.25, step=0.05)

    dataset_map = {
        "Spam Detection (Spam vs Ham)": config.BASE_DIR / "data" / "spam_dataset.csv",
        "Email Phishing Detection (Phishing vs Legitimate)": config.BASE_DIR / "data" / "phishing_dataset.csv"
    }
    target_csv = dataset_map[task_choice]

    if st.button("🚀 EXECUTE MODEL BENCHMARK", use_container_width=True):
        with st.spinner("Extracting TF-IDF features, training models, and generating confusion matrices..."):
            results = train_and_compare_models(
                csv_path=target_csv,
                task_name=task_choice,
                test_size=test_split
            )
            st.session_state["benchmark_results"] = results

    if "benchmark_results" in st.session_state:
        res = st.session_state["benchmark_results"]
        st.success(f"Benchmarked successfully on **{res['task_name']}**! ({res['dataset_size']} total samples).")

        col_nb, col_lr = st.columns(2)
        with col_nb:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #00f2fe;">
                    <div style="font-size: 15px; font-weight: 700; color: #00f2fe; margin-bottom: 8px;">MULTINOMIAL NAIVE BAYES</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div><span style="color:#94a3b8; font-size:12px;">Accuracy:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['naive_bayes']['accuracy']*100:.1f}%</span></div>
                        <div><span style="color:#94a3b8; font-size:12px;">F1-Score:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['naive_bayes']['f1_score']*100:.1f}%</span></div>
                        <div><span style="color:#94a3b8; font-size:12px;">Precision:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['naive_bayes']['precision']*100:.1f}%</span></div>
                        <div><span style="color:#94a3b8; font-size:12px;">Recall:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['naive_bayes']['recall']*100:.1f}%</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_lr:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #10b981;">
                    <div style="font-size: 15px; font-weight: 700; color: #10b981; margin-bottom: 8px;">LOGISTIC REGRESSION</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div><span style="color:#94a3b8; font-size:12px;">Accuracy:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['logistic_regression']['accuracy']*100:.1f}%</span></div>
                        <div><span style="color:#94a3b8; font-size:12px;">F1-Score:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['logistic_regression']['f1_score']*100:.1f}%</span></div>
                        <div><span style="color:#94a3b8; font-size:12px;">Precision:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['logistic_regression']['precision']*100:.1f}%</span></div>
                        <div><span style="color:#94a3b8; font-size:12px;">Recall:</span> <span style="color:#fff; font-weight:700; font-family:'JetBrains Mono';">{res['logistic_regression']['recall']*100:.1f}%</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Confusion Matrix Heatmaps (Side-by-Side)")
        if Path(res["confusion_matrix_image"]).exists():
            st.image(res["confusion_matrix_image"], use_column_width=True)

        st.markdown("---")
        st.subheader("🧪 Live Comparative Inference Prompt")
        test_input = st.text_input(
            "Enter sentence to test on both models:",
            value="URGENT: Your bank account was locked due to security anomalies. Re-verify password immediately.",
            key="bench_test_cyber"
        )
        if test_input:
            vec = res["vectorizer"]
            feat = vec.transform([test_input])
            nb_prob = res["nb_model"].predict_proba(feat)[0][1]
            lr_prob = res["lr_model"].predict_proba(feat)[0][1]

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.metric("Naive Bayes Probability", f"{nb_prob * 100:.1f}%")
                if nb_prob >= 0.5:
                    st.error("🚨 Naive Bayes: Malicious / Positive")
                else:
                    st.success("✅ Naive Bayes: Safe / Negative")
            with t_col2:
                st.metric("Logistic Regression Probability", f"{lr_prob * 100:.1f}%")
                if lr_prob >= 0.5:
                    st.error("🚨 Logistic Regression: Malicious / Positive")
                else:
                    st.success("✅ Logistic Regression: Safe / Negative")

# ---------------------------------------------------------
# VIEW 4: SQLITE FORENSIC LOGS
# ---------------------------------------------------------
elif nav_option == "📊 SQLite Forensic Logs":
    st.markdown("""
        <div class="hero-box">
            <h2 style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 0;">
                📊 SQLite Forensic Audit Trails
            </h2>
            <p style="color: #94a3b8; font-size: 13px; margin: 6px 0 0 0;">
                Relational record store for past threat scans, extracted links, and audit history.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("PURGE AUDIT HISTORY"):
            clear_all_history()
            st.success("Audit history purged.")
            st.rerun()

    tab_scans, tab_reports = st.tabs(["🔍 All Scans Audit Log", "🚨 Escalated Police Fraud Reports (1930)"])
    
    with tab_scans:
        scans = get_recent_scans(limit=100)
        if not scans:
            st.info("No recorded security scans found in SQLite database.")
        else:
            st.dataframe(pd.DataFrame(scans), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Inspect Detailed Forensic Record")
            scan_ids = [s["id"] for s in scans]
            chosen_id = st.selectbox("Select Scan ID to Inspect:", scan_ids)
            
            details = get_scan_details(chosen_id)
            if details:
                st.json({
                    "Audit_ID": details["id"],
                    "Timestamp": details["timestamp"],
                    "Sender": details["sender"],
                    "Subject": details["subject"],
                    "Verdict": details["verdict"],
                    "Threat_Level": details["threat_level"],
                    "Metrics": {
                        "Composite_Risk": details["composite_score"],
                        "Phishing_Score": details["phishing_score"],
                        "Spam_Score": details["spam_score"],
                        "Max_URL_Score": details["url_max_score"]
                    },
                    "Extracted_Links": details["urls"]
                })

    with tab_reports:
        fraud_reps = get_fraud_reports(limit=50)
        if not fraud_reps:
            st.info("No fraud incidents have been escalated yet. Reports logged when threat level > 70% will appear here.")
        else:
            st.write("Formal incident complaints logged for Cyber Crime Police (Helpline 1930) reporting:")
            st.dataframe(pd.DataFrame(fraud_reps), use_container_width=True)

# ---------------------------------------------------------
# VIEW 5: ARCHITECTURE SPECS
# ---------------------------------------------------------
elif nav_option == "⚙️ Architecture Specs":
    st.markdown("""
        <div class="hero-box">
            <h2 style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 0;">
                ⚙️ System Architecture & Technology Specifications
            </h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🛡️ Core Engine Architecture
    * **Frontend:** Streamlit Cyber-SOC HUD Interface with dark glassmorphism & responsive telemetry.
    * **Spam Filter:** TF-IDF n-grams (1, 2) + Multinomial Naive Bayes (Laplace smoothing).
    * **Phishing Filter:** TF-IDF n-grams (1, 2) + Logistic Regression (L2 regularization).
    * **URL Threat Analyzer:** 20-Feature structural/lexical extractor + Random Forest Classifier.
    * **Aggregator & Override Engine:** Multi-vector composite scoring with hard-override escalation for weaponized links.
    * **Explainability (XAI):** Linear model coefficient attributions ($w_i \cdot x_i$) & Naive Bayes log-odds difference.
    * **Audit Store:** SQLite relational database (`phish_guard.db`).
    """)

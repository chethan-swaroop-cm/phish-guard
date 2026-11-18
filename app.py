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
    page_title="PHISHGUARD // CYBER THREAT RADAR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# FUTURISTIC CYBERPUNK / SOC COMMAND STYLING (CSS INJECTION)
# ---------------------------------------------------------
FUTURISTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* Global Reset & Futuristic Dark Palette */
:root {
    --bg-primary: #070b14;
    --bg-secondary: #0d1527;
    --bg-glass: rgba(13, 21, 39, 0.7);
    --border-cyan: rgba(0, 242, 254, 0.25);
    --neon-cyan: #00f2fe;
    --neon-blue: #4facfe;
    --neon-red: #ff0055;
    --neon-amber: #f59e0b;
    --neon-green: #00ff88;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
}

/* Background canvas */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0d1527 0%, #070b14 70%, #03060a 100%) !important;
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #070b14;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 3px;
    border: 1px solid var(--border-cyan);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(10, 16, 30, 0.95) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.15) !important;
    backdrop-filter: blur(16px);
}

/* Futuristic HUD Header */
.hud-header {
    background: linear-gradient(90deg, rgba(0, 242, 254, 0.08) 0%, rgba(79, 172, 254, 0.03) 100%);
    border: 1px solid var(--border-cyan);
    border-left: 4px solid var(--neon-cyan);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 24px;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.08);
}
.hud-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #ffffff;
    text-transform: uppercase;
    font-family: 'Space Grotesk', sans-serif;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hud-subtitle {
    font-size: 13px;
    color: var(--neon-cyan);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Pulsing Online Radar Dot */
.pulse-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--neon-green);
    box-shadow: 0 0 10px var(--neon-green);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(0, 255, 136, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
}

/* Glassmorphism Metric Cards */
.cyber-card {
    background: rgba(13, 21, 39, 0.75);
    border: 1px solid rgba(0, 242, 254, 0.18);
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    margin-bottom: 16px;
}
.cyber-card:hover {
    border-color: rgba(0, 242, 254, 0.45);
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.15);
    transform: translateY(-2px);
}

/* Threat Status Banners */
.threat-banner-critical {
    background: linear-gradient(135deg, rgba(255, 0, 85, 0.18) 0%, rgba(255, 0, 85, 0.05) 100%);
    border: 1px solid rgba(255, 0, 85, 0.5);
    border-left: 6px solid #ff0055;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(255, 0, 85, 0.2);
    margin-bottom: 24px;
}
.threat-banner-high {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(245, 158, 11, 0.05) 100%);
    border: 1px solid rgba(245, 158, 11, 0.5);
    border-left: 6px solid #f59e0b;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
    margin-bottom: 24px;
}
.threat-banner-clean {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 255, 136, 0.03) 100%);
    border: 1px solid rgba(0, 255, 136, 0.4);
    border-left: 6px solid #00ff88;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.15);
    margin-bottom: 24px;
}

/* Monospace Chip Badges */
.badge-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
    display: inline-block;
}
.badge-critical {
    background: rgba(255, 0, 85, 0.15);
    color: #ff3377;
    border: 1px solid rgba(255, 0, 85, 0.4);
}
.badge-safe {
    background: rgba(0, 255, 136, 0.15);
    color: #00ff88;
    border: 1px solid rgba(0, 255, 136, 0.4);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.badge-cyan {
    background: rgba(0, 242, 254, 0.15);
    color: #00f2fe;
    border: 1px solid rgba(0, 242, 254, 0.4);
}

/* Custom Styled Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
    color: #050a15 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    box-shadow: 0 0 35px rgba(0, 242, 254, 0.6) !important;
    transform: scale(1.02) !important;
    color: #000000 !important;
}

/* Form Inputs */
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background: rgba(13, 21, 39, 0.8) !important;
    border: 1px solid rgba(0, 242, 254, 0.2) !important;
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.25) !important;
}

/* Metric text enhancements */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--text-secondary) !important;
}
</style>
"""
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)

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
    st.error(f"Error initializing models: {e}")
    st.stop()

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

# Sidebar - SOC Command Avionics
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="font-size: 38px;">🛡️</div>
            <div style="font-size: 20px; font-weight: 800; letter-spacing: 2px; color: #00f2fe; margin-top: 4px;">PHISHGUARD AI</div>
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #64748b; letter-spacing: 1px;">AUTONOMOUS THREAT RADAR v2.4</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(0,255,136,0.08); border: 1px solid rgba(0,255,136,0.3); border-radius: 6px; padding: 8px 12px; margin-bottom: 16px;">
            <span class="pulse-indicator"></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #00ff88; font-weight: 600;">SOC NODE ONLINE // ACTIVE</span>
        </div>
    """, unsafe_allow_html=True)

    nav_option = st.radio(
        "OPERATIONAL COMMAND",
        [
            "🛡️ Cyber Threat Scanner",
            "🔗 Tactical URL Matrix",
            "⚖️ Model Benchmarking (ML)",
            "📊 SQLite Forensic Logs",
            "⚙️ Architecture Specs"
        ]
    )
    
    st.markdown("---")
    st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 12px; color: #00f2fe; margin-bottom: 8px;'>TELEMETRY COUNTERS</div>", unsafe_allow_html=True)
    stats = get_system_stats()
    c_s1, c_s2 = st.columns(2)
    c_s1.metric("Scans Logged", stats["total_scans"])
    c_s2.metric("Threats Quarantined", stats["phishing_count"])
    c_s3, c_s4 = st.columns(2)
    c_s3.metric("Spam Trapped", stats["spam_count"])
    c_s4.metric("Malicious URLs", stats["malicious_urls"])

    st.markdown("---")
    st.caption("Engine: TF-IDF + Naive Bayes + Logistic Regression + Random Forest + XAI Attribution")

# ---------------------------------------------------------
# VIEW 1: CYBER THREAT SCANNER
# ---------------------------------------------------------
if nav_option == "🛡️ Cyber Threat Scanner":
    # Futuristic HUD Banner
    st.markdown(f"""
        <div class="hud-header">
            <div class="hud-title">
                <span>🛡️</span> EMAIL THREAT DETECTION & COMBINED FORENSICS
            </div>
            <div class="hud-subtitle">
                TRI-VECTOR ANALYSIS: LINGUISTIC PHISHING // PROMOTIONAL SPAM // MULTI-LINK STRUCTURAL ANOMALIES
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Simulation preset selector
    selected_sample = st.selectbox(
        "⚡ Quick-Load Threat Injection Scenario:",
        list(SAMPLE_EMAILS.keys())
    )
    default_subj = SAMPLE_EMAILS[selected_sample]["subject"]
    default_sender = SAMPLE_EMAILS[selected_sample]["sender"]
    default_body = SAMPLE_EMAILS[selected_sample]["body"]

    with st.form("cyber_scan_form"):
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            input_sender = st.text_input("Origin Header / Sender Address:", value=default_sender, placeholder="e.g., auth@security-portal.xyz")
        with col_m2:
            input_subject = st.text_input("Message Subject Line:", value=default_subj, placeholder="e.g., URGENT: Credentials Suspended")
            
        input_body = st.text_area(
            "Raw Payload / Email Content (Plaintext or HTML):",
            value=default_body,
            height=220,
            placeholder="Paste raw email body or MIME payload here..."
        )
        
        scan_btn = st.form_submit_button("⚡ EXECUTE MULTI-VECTOR THREAT AUDIT", use_container_width=True)

    if scan_btn:
        if not input_body.strip():
            st.warning("⚠️ No email payload provided. Enter content or select a preset scenario.")
        else:
            with st.spinner("Decoding payload, extracting embedded hyperlinks, running TF-IDF models..."):
                # 1. Feature 1: Extract All Embedded URLs
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

            # --- COMBINED THREAT REPORT HUD BANNER ---
            st.markdown("---")
            threat_level = report["threat_level"]
            
            if threat_level == "CRITICAL":
                banner_class = "threat-banner-critical"
                icon = "🚨"
                verdict_color = "#ff0055"
            elif threat_level in ("HIGH", "MEDIUM"):
                banner_class = "threat-banner-high"
                icon = "⚠️"
                verdict_color = "#f59e0b"
            else:
                banner_class = "threat-banner-clean"
                icon = "✅"
                verdict_color = "#00ff88"

            st.markdown(f"""
                <div class="{banner_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: {verdict_color}; letter-spacing: 2px;">
                                // FORENSIC VERDICT // STATUS: {threat_level}
                            </div>
                            <div style="font-size: 24px; font-weight: 800; color: #ffffff; margin-top: 4px;">
                                {icon} {report["verdict"]}
                            </div>
                        </div>
                        <div style="text-align: right; margin-top: 8px;">
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8;">COMPOSITE RISK INDEX</div>
                            <div style="font-size: 32px; font-weight: 900; font-family: 'JetBrains Mono', monospace; color: {verdict_color};">
                                {report['scores']['composite_percent']}%
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 14px; color: #e2e8f0;">
                        <strong>DEFENSIVE ACTION PROTOCOL:</strong> {report["action_advice"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Four Cyber Score Cards
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"""
                    <div class="cyber-card">
                        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">COMPOSITE THREAT</div>
                        <div style="font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {verdict_color}; margin-top: 6px;">
                            {report['scores']['composite_percent']}%
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Overall Attack Potential</div>
                    </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                    <div class="cyber-card">
                        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">PHISHING INTENT</div>
                        <div style="font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #00f2fe; margin-top: 6px;">
                            {round(report['scores']['phishing_score'] * 100, 1)}%
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">TF-IDF + Logistic Reg</div>
                    </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                    <div class="cyber-card">
                        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">BULK SPAM INDEX</div>
                        <div style="font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #f59e0b; margin-top: 6px;">
                            {round(report['scores']['spam_score'] * 100, 1)}%
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">TF-IDF + Naive Bayes</div>
                    </div>
                """, unsafe_allow_html=True)
            with sc4:
                st.markdown(f"""
                    <div class="cyber-card">
                        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">MAX EMBEDDED URL RISK</div>
                        <div style="font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {'#ff0055' if report['scores']['url_max_score']>=0.5 else '#00ff88'}; margin-top: 6px;">
                            {round(report['scores']['url_max_score'] * 100, 1)}%
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">{report['num_urls']} Hyperlinks Analyzed</div>
                    </div>
                """, unsafe_allow_html=True)

            # --- STANDOUT FEATURE 1: EMBEDDED URL TACTICAL AUDIT ---
            st.markdown("---")
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 18px; font-weight: 700; color: #ffffff;">
                        🔗 EMBEDDED HYPERLINK DOSSIER ({report['num_urls']} LINKS DETECTED)
                    </div>
                    <span class="badge-mono badge-cyan">Feature 1 // Multi-Link Deep Audit</span>
                </div>
            """, unsafe_allow_html=True)

            if report["num_urls"] == 0:
                st.info("No embedded URLs located in the payload.")
            else:
                for idx, u in enumerate(url_eval_results, start=1):
                    risk_pct = round(u['risk_score'] * 100, 1)
                    is_susp = u['is_suspicious']
                    card_border = "rgba(255, 0, 85, 0.4)" if is_susp else "rgba(0, 255, 136, 0.3)"
                    badge_markup = '<span class="badge-mono badge-critical">MALICIOUS</span>' if is_susp else '<span class="badge-mono badge-safe">CLEAN</span>'
                    
                    with st.expander(f"Link #{idx} // [{risk_pct}% RISK] {u['url']}", expanded=is_susp):
                        col_u1, col_u2 = st.columns([1, 2])
                        with col_u1:
                            st.markdown(f"**Status:** {badge_markup}", unsafe_allow_html=True)
                            st.markdown(f"**Risk Score:** `{risk_pct}%`")
                            st.markdown(f"**Target URL:** `{u['url']}`")
                        with col_u2:
                            st.markdown("**Structural & Lexical Flags Detected:**")
                            if u["reasons"]:
                                for r in u["reasons"]:
                                    st.markdown(f"- ⚠️ **Flag:** {r}")
                            else:
                                st.markdown("- ✅ *Passed all standard lexical filters (no suspicious TLD, no IP hostname, standard entropy).*")

            # --- STANDOUT FEATURE 3: EXPLAINABLE AI (XAI) PANEL ---
            st.markdown("---")
            st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 18px; font-weight: 700; color: #ffffff;">
                        💡 EXPLAINABLE AI (XAI): NEURAL & STATISTICAL ATTRIBUTION
                    </div>
                    <span class="badge-mono badge-cyan">Feature 3 // Feature-Based Explainability</span>
                </div>
            """, unsafe_allow_html=True)

            xai_tab1, xai_tab2, xai_tab3 = st.tabs([
                "🎣 Phishing Drivers (Logistic Regression Weights)",
                "📧 Spam Triggers (Naive Bayes Log-Odds)",
                "🌐 Lexical & Domain Anomaly Indicators"
            ])

            with xai_tab1:
                st.caption(f"Reasoning: {report['explainability']['text_analysis']['phish_summary']}")
                col_pt1, col_pt2 = st.columns(2)
                with col_pt1:
                    st.markdown("##### 🚨 Top Phishing Contributors (Positive Weights)")
                    pt = report["explainability"]["text_analysis"]["phish_triggers"]
                    if pt:
                        for item in pt:
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; background: rgba(255,0,85,0.08); border-left: 3px solid #ff0055; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #fff;">{item['token']}</span>
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #ff0055; font-weight: bold;">+{item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No significant phishing n-grams detected.")
                with col_pt2:
                    st.markdown("##### 🛡️ Top Benign Indicators (Negative Weights)")
                    si = report["explainability"]["text_analysis"]["safe_indicators"]
                    if si:
                        for item in si:
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; background: rgba(0,255,136,0.08); border-left: 3px solid #00ff88; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #fff;">{item['token']}</span>
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #00ff88; font-weight: bold;">-{item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No strong legitimate tokens countered the classification.")

            with xai_tab2:
                st.caption(f"Reasoning: {report['explainability']['text_analysis']['spam_summary']}")
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st.markdown("##### 📢 Promotional Spam Tokens (Log-Odds)")
                    st_trig = report["explainability"]["text_analysis"]["spam_triggers"]
                    if st_trig:
                        for item in st_trig:
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #fff;">{item['token']}</span>
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #f59e0b; font-weight: bold;">Δ {item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No active promotional vocabulary triggers found.")
                with col_st2:
                    st.markdown("##### 💬 Standard Correlated Ham Vocabulary")
                    ham_ind = report["explainability"]["text_analysis"]["ham_indicators"]
                    if ham_ind:
                        for item in ham_ind:
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; background: rgba(0,242,254,0.08); border-left: 3px solid #00f2fe; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #fff;">{item['token']}</span>
                                    <span style="font-family: 'JetBrains Mono', monospace; color: #00f2fe; font-weight: bold;">{item['weight']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No strong ham indicators found.")

            with xai_tab3:
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
        <div class="hud-header">
            <div class="hud-title"><span>🔗</span> STANDALONE URL THREAT MATRIX</div>
            <div class="hud-subtitle">EXTRACTS 20+ LEXICAL, STRUCTURAL & SHANNON ENTROPY ANOMALIES IN REAL-TIME</div>
        </div>
    """, unsafe_allow_html=True)

    target_url = st.text_input(
        "Enter Target URL to Probe:",
        value="http://192.168.1.100/paypal-security/login.php?cmd=verify",
        placeholder="https://suspicious-domain.xyz/auth"
    )

    if st.button("PROBE URL MATRIX", use_container_width=True):
        if not target_url.strip():
            st.warning("Please supply a URL string.")
        else:
            with st.spinner("Extracting structural features and scoring heuristic weights..."):
                u_res = url_detector.predict_url(target_url.strip())
                f_dict = extract_url_feature_dict(target_url.strip())
                
            risk_pct = round(u_res['risk_score'] * 100, 1)
            is_susp = u_res["is_suspicious"]
            
            c_u1, c_u2 = st.columns([1, 2])
            with c_u1:
                st.markdown(f"""
                    <div class="cyber-card" style="text-align: center;">
                        <div style="font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">CALCULATED MALICIOUS PROBABILITY</div>
                        <div style="font-size: 42px; font-weight: 900; font-family: 'JetBrains Mono', monospace; color: {'#ff0055' if is_susp else '#00ff88'}; margin: 12px 0;">
                            {risk_pct}%
                        </div>
                        <div>
                            {'<span class="badge-mono badge-critical">THREAT VERDICT: MALICIOUS</span>' if is_susp else '<span class="badge-mono badge-safe">THREAT VERDICT: BENIGN</span>'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with c_u2:
                st.markdown("### 🚩 Heuristic Anomaly Indicators")
                if u_res["reasons"]:
                    for r in u_res["reasons"]:
                        st.markdown(f"""
                            <div style="background: rgba(255,0,85,0.08); border-left: 3px solid #ff0055; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px; font-family: 'Space Grotesk', sans-serif;">
                                ⚠️ {r}
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ Clean URL. No abnormal domain structures, ports, or deceptive markers found.")

            st.markdown("---")
            st.subheader("📊 20 Granular Lexical & Structural Metrics")
            df_feat = pd.DataFrame(list(f_dict.items()), columns=["Metric Key", "Computed Value"])
            st.dataframe(df_feat, use_container_width=True)

# ---------------------------------------------------------
# VIEW 3: MODEL BENCHMARKING (ML)
# ---------------------------------------------------------
elif nav_option == "⚖️ Model Benchmarking (ML)":
    st.markdown("""
        <div class="hud-header">
            <div class="hud-title"><span>⚖️</span> ALGORITHM BENCHMARK & COMPARISON</div>
            <div class="hud-subtitle">TF-IDF VECTORIZATION // MULTINOMIAL NAIVE BAYES VS. LOGISTIC REGRESSION // CONFUSION MATRICES</div>
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
        st.success(f"Benchmarked successfully on **{res['task_name']}**! ({res['dataset_size']} total samples, {res['train_size']} train, {res['test_size']} test).")

        # Metric KPI cards
        col_nb, col_lr = st.columns(2)
        with col_nb:
            st.markdown(f"""
                <div class="cyber-card" style="border-top: 3px solid #00f2fe;">
                    <div style="font-size: 16px; font-weight: 700; color: #00f2fe; margin-bottom: 10px;">MULTINOMIAL NAIVE BAYES</div>
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
                <div class="cyber-card" style="border-top: 3px solid #00ff88;">
                    <div style="font-size: 16px; font-weight: 700; color: #00ff88; margin-bottom: 10px;">LOGISTIC REGRESSION</div>
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
            "Enter live sentence to test on both models:",
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
        <div class="hud-header">
            <div class="hud-title"><span>📊</span> SQLITE FORENSIC AUDIT TRAILS</div>
            <div class="hud-subtitle">RELATIONAL RECORD STORE // SCAN REPLAY & HISTORICAL ANOMALY TRACKING</div>
        </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("PURGE AUDIT HISTORY"):
            clear_all_history()
            st.success("Audit history purged.")
            st.rerun()

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

# ---------------------------------------------------------
# VIEW 5: ARCHITECTURE SPECS
# ---------------------------------------------------------
elif nav_option == "⚙️ Architecture Specs":
    st.markdown("""
        <div class="hud-header">
            <div class="hud-title"><span>⚙️</span> SYSTEM ARCHITECTURE SPECIFICATIONS</div>
            <div class="hud-subtitle">AUTONOMOUS CYBER DEFENSE PIPELINE SPECIFICATION</div>
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

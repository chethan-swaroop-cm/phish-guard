import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from models.spam_detector import SpamDetector
from models.phishing_detector import PhishingDetector
from models.url_detector import URLDetector
from utils.feature_extraction import extract_url_feature_vector
import config

def get_seed_spam_data() -> Tuple[List[str], List[int]]:
    """Curated representative seed dataset for Spam (1) vs Ham (0)."""
    ham_samples = [
        "Hi Team, attached is the revised agenda for our quarterly strategy sync tomorrow at 10 AM.",
        "Could you please review the pull request for the authentication module when you have a moment?",
        "Your weekly sprint summary is ready. 14 tickets were resolved and 3 were carried over.",
        "Let's grab lunch around 12:30 at the campus cafe if you're free today.",
        "Here are the meeting notes and action items from yesterday's product design review.",
        "The scheduled database migration was completed successfully with zero downtime reported.",
        "Please find the invoice for the cloud hosting services for the month of October.",
        "Reminder: The company-wide all-hands meeting will be broadcast live tomorrow at 3 PM EST.",
        "Thanks for sending over the project timeline. I have shared my feedback directly on Google Docs.",
        "Can you send me the latest deck before our client presentation this afternoon?",
        "The automated build test pipeline finished in 4 minutes with all test suites passing.",
        "Here is the monthly engineering newsletter detailing our migration to Python 3.11."
    ]
    
    spam_samples = [
        "CONGRATULATIONS! You have been selected as the grand winner of the 2026 International Cash Lottery! Claim now!",
        "Double your Bitcoin in 24 hours guaranteed! Fast payouts, no risk, invest today with certified brokers.",
        "URGENT: Inherited funds of $10,500,000 USD awaiting wire transfer to your account. Send your bank details immediately.",
        "Dear Friend, I am a former government minister with confidential funds seeking your urgent partnership for transfer.",
        "Exclusive offer! Buy prescription medications, weight loss supplements and vitamins at 90% discount without prescription!",
        "Pre-approved loan up to $500,000 with 0% interest for the first 12 months. Act now, limited time offer!",
        "Earn $5,000 daily from home doing simple online surveys. No experience required. Start now!",
        "Clear all your credit card debts instantly with our government-approved relief program. Apply today!",
        "You won a $1,000 Walmart gift card! Click here immediately to claim before this promotion expires!",
        "Guaranteed lottery prize payout waiting for confirmation. Reply with your passport and home address."
    ]
    
    texts = ham_samples + spam_samples
    labels = [0] * len(ham_samples) + [1] * len(spam_samples)
    return texts, labels

def get_seed_phishing_data() -> Tuple[List[str], List[int]]:
    """Curated representative seed dataset for Phishing (1) vs Safe (0)."""
    safe_samples = [
        "Your Amazon order #402-99812 has shipped and is estimated to arrive on Thursday.",
        "Hi John, here is the updated contract with the legal team's redlines for your review.",
        "Monthly GitHub notifications: 5 new issues assigned to your team repositories.",
        "Your payment for the subscription was processed. Thank you for continuing your membership.",
        "Attached are the flight confirmation details and hotel itinerary for the conference.",
        "Reminder: Flu shot clinic available on the 2nd floor today from 9 AM to 3 PM.",
        "The documentation site has been updated with the latest API endpoints and examples.",
        "Hi Alex, did you get a chance to look at the budget proposal for the next fiscal year?",
        "Welcome to the workspace! You can set up your profile and notifications in preferences.",
        "Thank you for contacting customer support. Your ticket #88210 has been resolved."
    ]
    
    phishing_samples = [
        "URGENT: Your PayPal account has been suspended due to unauthorized login attempts. Verify identity within 24 hours.",
        "Security Alert: Unusual sign-in activity detected from Moscow, Russia. If this wasn't you, click here to secure your account.",
        "Microsoft 365 Notice: Your password expires in 2 hours. Keep your current password by verifying your credentials now.",
        "IRS Urgent Notice: You have an unclaimed tax refund of $1,420.50. Submit your SSN and debit card details to receive deposit.",
        "Bank of America Security Dept: We noticed irregular wire activity. Access restricted. Confirm account details immediately.",
        "IT Help Desk Alert: Critical mail server upgrade in progress. Failure to validate your login credentials will terminate your inbox.",
        "Apple ID Notice: Your account has been locked for security reasons. Click below to verify your billing information immediately.",
        "DocuSign: Important confidential document waiting for signature. Log in with your email password to decrypt and view file.",
        "Payroll Alert: Direct deposit information mismatch. Update your bank routing number before Friday to ensure on-time payment.",
        "Your Netflix membership is on hold. We were unable to process your payment. Update your credit card now to resume streaming."
    ]
    
    texts = safe_samples + phishing_samples
    labels = [0] * len(safe_samples) + [1] * len(phishing_samples)
    return texts, labels

def get_seed_url_data() -> Tuple[List[str], List[int]]:
    """Curated representative seed dataset for Malicious/Phishing URLs (1) vs Benign URLs (0)."""
    benign_urls = [
        "https://www.google.com",
        "https://github.com/torvalds/linux",
        "https://en.wikipedia.org/wiki/Computer_security",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://stackoverflow.com/questions/5074212/how-to-write-a-regex",
        "https://www.microsoft.com/en-us/windows",
        "https://www.amazon.com/dp/B08N5WRWNW",
        "https://www.nytimes.com/section/technology",
        "https://streamlit.io/gallery",
        "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html",
        "https://www.cloudflare.com/learning/access-management/what-is-zero-trust/",
        "https://www.apple.com/macos/sonoma/"
    ]
    
    malicious_urls = [
        "http://192.168.1.105/paypal/login.php?cmd=verify",
        "http://security-update-paypal.xyz/account/signin",
        "https://appleid.apple.com.ssl-verify-account.top/login",
        "http://netflix-billing-update-suspicious.buzz/login",
        "http://bit.ly/3xY7q9P-fake-login",
        "http://user:pass@evil-phishing-host.tk/banking",
        "http://https-wellsfargo-security.xyz:8080/secure/auth.html",
        "http://login.microsoftonline.com.account-recovery-alert.click/portal",
        "http://185.220.101.44/dhl/track-shipment-confirm-id",
        "http://irs-tax-refund-government-portal.kim/verify-ssn.php",
        "https://secure-chase-online-banking-token.work/login"
    ]
    
    urls = benign_urls + malicious_urls
    labels = [0] * len(benign_urls) + [1] * len(malicious_urls)
    return urls, labels

def train_and_save_all_models():
    """Trains and serializes all 3 models to config.SAVED_MODELS_DIR."""
    print("Training Spam Detector (TF-IDF + Naive Bayes)...")
    spam_texts, spam_labels = get_seed_spam_data()
    spam_detector = SpamDetector().fit(spam_texts, spam_labels)
    spam_detector.save(config.SPAM_MODEL_PATH)
    print(f"Spam Detector saved to {config.SPAM_MODEL_PATH}")

    print("Training Phishing Detector (TF-IDF + Logistic Regression)...")
    phish_texts, phish_labels = get_seed_phishing_data()
    phish_detector = PhishingDetector().fit(phish_texts, phish_labels)
    phish_detector.save(config.PHISHING_MODEL_PATH)
    print(f"Phishing Detector saved to {config.PHISHING_MODEL_PATH}")

    print("Training URL Detector (Feature Extraction + Random Forest)...")
    raw_urls, url_labels = get_seed_url_data()
    url_vectors = [extract_url_feature_vector(u) for u in raw_urls]
    url_detector = URLDetector().fit(url_vectors, url_labels)
    url_detector.save(config.URL_MODEL_PATH)
    print(f"URL Detector saved to {config.URL_MODEL_PATH}")

    print("All baseline models successfully trained and serialized!")

if __name__ == "__main__":
    train_and_save_all_models()

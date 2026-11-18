from typing import Dict, Any, List
import config
from utils.explainability import build_explainability_report

def synthesize_threat_report(
    subject: str,
    sender: str,
    body: str,
    spam_res: Dict[str, Any],
    phish_res: Dict[str, Any],
    url_results: List[Dict[str, Any]],
    spam_exp: Dict[str, Any],
    phish_exp: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standout Feature 2: Combined Threat Report.
    Synthesizes text spam classification, phishing classification,
    and individual embedded link analyses into a unified defense dossier.
    """
    spam_prob = float(spam_res.get("spam_probability", 0.0))
    phish_prob = float(phish_res.get("phishing_probability", 0.0))
    
    # URL metrics
    num_urls = len(url_results)
    if num_urls > 0:
        url_scores = [float(u.get("risk_score", 0.0)) for u in url_results]
        url_max = max(url_scores)
        url_avg = sum(url_scores) / num_urls
        # Standard weighted composite
        raw_composite = (
            config.WEIGHT_PHISHING * phish_prob +
            config.WEIGHT_URL * url_max +
            config.WEIGHT_SPAM * spam_prob
        )
    else:
        url_max = 0.0
        url_avg = 0.0
        # Re-weight text signals proportionally when no URLs are present
        total_text_w = config.WEIGHT_PHISHING + config.WEIGHT_SPAM
        w_phish = config.WEIGHT_PHISHING / total_text_w
        w_spam = config.WEIGHT_SPAM / total_text_w
        raw_composite = (w_phish * phish_prob) + (w_spam * spam_prob)

    composite_score = round(min(1.0, max(0.0, raw_composite)), 4)
    composite_percent = round(composite_score * 100, 1)

    # Multi-hazard Hard Overrides
    has_weaponized_url = any(u.get("risk_score", 0.0) >= 0.70 for u in url_results)
    
    if has_weaponized_url:
        threat_level = "CRITICAL"
        verdict = "CRITICAL: MALICIOUS LINK DETECTED (PHISHING)"
        action_advice = "DO NOT CLICK ANY LINKS. Quarantine email immediately and report domain to SOC/Security team."
        composite_percent = max(composite_percent, 88.0)
    elif phish_prob >= 0.70:
        threat_level = "HIGH"
        verdict = "HIGH RISK: CREDENTIAL PHISHING ATTACK"
        action_advice = "Do not reply or provide credentials. High probability of social engineering impersonation."
        composite_percent = max(composite_percent, 75.0)
    elif spam_prob >= 0.65 and url_max < 0.50:
        threat_level = "MEDIUM"
        verdict = "SUSPICIOUS: SPAM / PROMOTIONAL"
        action_advice = "Unsolicited bulk email. Safe to delete or move to junk folder."
    elif composite_percent >= 45.0:
        threat_level = "MEDIUM"
        verdict = "MODERATE RISK: SUSPICIOUS CONTENT"
        action_advice = "Exercise caution. Verify sender address before interacting."
    else:
        threat_level = "LOW"
        verdict = "CLEAN: LEGITIMATE MESSAGE"
        action_advice = "No significant threats or anomalies detected. Safe to open."

    # Build XAI dossier
    xai_dossier = build_explainability_report(
        spam_explanation=spam_exp,
        phishing_explanation=phish_exp,
        url_results=url_results,
        text_content=body
    )

    return {
        "metadata": {
            "subject": subject or "(No Subject)",
            "sender": sender or "(Unknown Sender)",
            "body_snippet": body[:300] + ("..." if len(body) > 300 else "")
        },
        "scores": {
            "spam_score": round(spam_prob, 4),
            "phishing_score": round(phish_prob, 4),
            "url_max_score": round(url_max, 4),
            "url_avg_score": round(url_avg, 4),
            "composite_score": composite_score,
            "composite_percent": composite_percent
        },
        "threat_level": threat_level,
        "verdict": verdict,
        "action_advice": action_advice,
        "num_urls": num_urls,
        "urls": url_results,
        "explainability": xai_dossier
    }

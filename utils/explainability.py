from typing import Dict, Any, List

def build_explainability_report(
    spam_explanation: Dict[str, Any],
    phishing_explanation: Dict[str, Any],
    url_results: List[Dict[str, Any]],
    text_content: str
) -> Dict[str, Any]:
    """
    Constructs an aggregated Explainable AI (XAI) dossier detailing:
    1. Text spam linguistic triggers (Naive Bayes log-odds)
    2. Text phishing social-engineering triggers (Logistic Regression weights)
    3. Structural and lexical red flags for each embedded URL
    """
    
    # 1. Spam Reasoning
    spam_triggers = spam_explanation.get("spam_triggers", [])
    ham_indicators = spam_explanation.get("ham_indicators", [])
    
    if spam_triggers:
        top_spam_words = [f"'{t['token']}' (+{t['weight']:.2f})" for t in spam_triggers[:4]]
        spam_summary = f"Identified promotional/spam phrase patterns: {', '.join(top_spam_words)}."
    else:
        spam_summary = "No prominent promotional spam phrases detected in body text."

    # 2. Phishing Reasoning
    phish_triggers = phishing_explanation.get("phishing_triggers", [])
    safe_indicators = phishing_explanation.get("safe_indicators", [])
    
    if phish_triggers:
        top_phish_words = [f"'{t['token']}' (+{t['weight']:.2f})" for t in phish_triggers[:4]]
        phish_summary = f"Urgency or credential harvesting phrasing detected: {', '.join(top_phish_words)}."
    else:
        phish_summary = "Text does not exhibit aggressive credential harvesting or impersonation language."

    # 3. URL Lexical & Structural Reasoning
    url_summaries = []
    total_suspicious_urls = 0
    for u in url_results:
        u_url = u.get("url", "")
        reasons = u.get("reasons", [])
        risk = u.get("risk_score", 0.0)
        is_susp = u.get("is_suspicious", False)
        if is_susp:
            total_suspicious_urls += 1
            
        url_summaries.append({
            "url": u_url,
            "risk_score": risk,
            "is_suspicious": is_susp,
            "reasons": reasons if reasons else ["URL structure conforms to standard conventions."]
        })

    return {
        "text_analysis": {
            "spam_summary": spam_summary,
            "spam_triggers": spam_triggers,
            "ham_indicators": ham_indicators,
            "phish_summary": phish_summary,
            "phish_triggers": phish_triggers,
            "safe_indicators": safe_indicators
        },
        "url_analysis": {
            "total_urls_scanned": len(url_results),
            "suspicious_urls_count": total_suspicious_urls,
            "url_breakdown": url_summaries
        }
    }

import math
import re
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple
import config

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_percent_signs",
    "num_digits",
    "is_ip",
    "num_subdomains",
    "has_suspicious_tld",
    "is_shortener",
    "keyword_count",
    "entropy",
    "has_port",
    "has_https_in_domain"
]

def calculate_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy to detect algorithmically generated or obfuscated domains."""
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())

def extract_url_feature_dict(url: str) -> Dict[str, Any]:
    """Extracts granular lexical and structural features from a single URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")
        
    netloc = parsed.netloc or ""
    clean_domain = netloc.split(":")[0].lower()
    path = parsed.path or ""
    
    # Check IP host
    is_ip = 1 if IP_PATTERN.match(clean_domain) else 0
    
    # Subdomains & TLD
    domain_parts = clean_domain.split(".")
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    num_subdomains = max(0, len(domain_parts) - 2) if not is_ip else 0
    has_suspicious_tld = 1 if tld in config.SUSPICIOUS_TLDS else 0
    
    # Shortener check
    is_shortener = 1 if clean_domain in config.URL_SHORTENERS else 0
    
    # Keywords
    url_lower = url.lower()
    keyword_count = sum(1 for kw in config.SUSPICIOUS_URL_KEYWORDS if kw in url_lower)
    
    # Port
    has_port = 1 if ":" in netloc and not clean_domain.startswith("[") else 0
    
    # HTTPS deception in domain (e.g., http://https-paypal-login.com)
    has_https_in_domain = 1 if ("https" in clean_domain or "http" in clean_domain) and parsed.scheme == "http" else 0
    
    entropy = round(calculate_shannon_entropy(clean_domain), 3)
    
    features = {
        "url_length": len(url),
        "domain_length": len(clean_domain),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_percent_signs": url.count("%"),
        "num_digits": sum(c.isdigit() for c in url),
        "is_ip": is_ip,
        "num_subdomains": num_subdomains,
        "has_suspicious_tld": has_suspicious_tld,
        "is_shortener": is_shortener,
        "keyword_count": keyword_count,
        "entropy": entropy,
        "has_port": has_port,
        "has_https_in_domain": has_https_in_domain
    }
    
    return features

def extract_url_feature_vector(url: str) -> List[float]:
    """Returns vector aligned with FEATURE_NAMES for ML inference."""
    f_dict = extract_url_feature_dict(url)
    return [float(f_dict[name]) for name in FEATURE_NAMES]

def explain_url_heuristics(url: str) -> List[str]:
    """
    Standout Feature 3 (Explainable AI - URLs):
    Translates detected lexical anomalies into human-interpretable security warnings.
    """
    f = extract_url_feature_dict(url)
    parsed = urlparse(url)
    clean_domain = parsed.netloc.split(":")[0].lower()
    tld = clean_domain.split(".")[-1] if "." in clean_domain else ""
    
    warnings = []
    
    if f["is_ip"] == 1:
        warnings.append(f"Host uses a raw IP address ({clean_domain}) instead of a reputable domain name.")
        
    if f["num_at_symbols"] > 0:
        warnings.append("URL contains '@' character, frequently used to disguise the real destination host.")
        
    if f["is_shortener"] == 1:
        warnings.append(f"URL uses known shortening service ('{clean_domain}'), obfuscating the final landing page.")
        
    if f["has_suspicious_tld"] == 1:
        warnings.append(f"Uses high-abuse top-level domain (.{tld}) commonly seen in phishing and spam campaigns.")
        
    if f["num_subdomains"] >= 3:
        warnings.append(f"Contains excessive subdomain nesting ({f['num_subdomains']} levels), typical of phishing impersonation.")
        
    if f["has_https_in_domain"] == 1:
        warnings.append("Misleading 'https' token embedded inside unencrypted HTTP domain name.")
        
    if f["keyword_count"] >= 2:
        found_kws = [kw for kw in config.SUSPICIOUS_URL_KEYWORDS if kw in url.lower()]
        warnings.append(f"Contains high-risk credential/banking keywords: {', '.join(found_kws[:3])}.")
        
    if f["url_length"] > 100:
        warnings.append(f"Unusually long URL ({f['url_length']} characters) often used to conceal token injection or redirect chains.")
        
    if f["entropy"] > 3.9:
        warnings.append(f"Elevated domain entropy ({f['entropy']:.2f}) suggests algorithmic or randomized domain generation.")
        
    if f["has_port"] == 1:
        warnings.append("Uses non-standard web port in destination host.")
        
    if f["num_hyphens"] >= 3:
        warnings.append(f"High hyphen count ({f['num_hyphens']}) in URL, frequently found in typosquatting attempts.")
        
    return warnings

import re
from urllib.parse import urlparse
from typing import List, Dict, Any

# Regex to capture standard URLs, www-prefixed links, and defanged formats
URL_PATTERN = re.compile(
    r"""(?xi)
    \b
    (?:
      # Defanged or normal scheme
      (?:https?|hxxps?|ftp)://
      |
      # Common www prefix
      www\d{0,3}\.
    )
    [^\s()<>{}\[\]"\']+
    """,
    re.VERBOSE
)

# HTML href pattern
HREF_PATTERN = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)

def defang_url_normalize(raw_url: str) -> str:
    """Restores defanged indicators often used in security research or phishing obfuscation."""
    url = raw_url.strip()
    # Replace defanged schemes
    url = re.sub(r"^hxxp", "http", url, flags=re.IGNORECASE)
    url = re.sub(r"^hxxps", "https", url, flags=re.IGNORECASE)
    # Replace defanged dots
    url = re.sub(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", ".", url, flags=re.IGNORECASE)
    # Replace defanged colons
    url = re.sub(r"\[:\]|\(:\)", ":", url)
    
    # Strip trailing punctuation commonly picked up from sentences
    url = url.rstrip(".,;:)>]\"'")
    
    # Prepend scheme if starting with www
    if url.lower().startswith("www."):
        url = "http://" + url
        
    return url

def normalize_email_text(raw_text: str) -> str:
    """Pre-normalizes text by expanding common security defanging conventions."""
    if not raw_text:
        return ""
    text = raw_text
    # Defang dots: [.] (.) {.} [dot] (dot)
    text = re.sub(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", ".", text, flags=re.IGNORECASE)
    # Defang schemes: hxxp[://], hxxps[://], hxxp[:]//, hxxp://
    text = re.sub(
        r"\bhxxp(s)?(?:\[:\/\/\]|\[:\]\/\/|:\/\/|\[:\]\[\/\/\])",
        lambda m: "https://" if m.group(1) else "http://",
        text,
        flags=re.IGNORECASE
    )
    return text

def extract_urls(text: str) -> List[str]:
    """
    Extracts all embedded URLs from plaintext or HTML email content.
    Automatically handles href tags, plaintext links, and defanged links.
    Returns a deduplicated list of normalized URLs.
    """
    if not text:
        return []
    
    # 1. Pre-process to normalize defanged indicators
    cleaned_text = normalize_email_text(text)
    
    found_urls = []
    
    # 2. Extract from HTML href attributes
    for match in HREF_PATTERN.findall(cleaned_text):
        normalized = defang_url_normalize(match)
        if normalized.startswith(("http://", "https://", "ftp://")):
            found_urls.append(normalized)
            
    # 3. Extract from raw regex pattern
    for match in URL_PATTERN.findall(cleaned_text):
        normalized = defang_url_normalize(match)
        if normalized.startswith(("http://", "https://", "ftp://")):
            found_urls.append(normalized)
            
    # 4. Deduplicate while preserving order
    unique_urls = []
    seen = set()
    for u in found_urls:
        u_lower = u.lower()
        if u_lower not in seen:
            seen.add(u_lower)
            unique_urls.append(u)
            
    return unique_urls

def parse_url_components(url: str) -> Dict[str, str]:
    """Extracts scheme, netloc/domain, path, params, query, and fragment."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        # Strip port if present for clean domain
        clean_domain = domain.split(":")[0] if ":" in domain else domain
        return {
            "url": url,
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "domain": clean_domain.lower(),
            "path": parsed.path,
            "query": parsed.query
        }
    except Exception:
        return {
            "url": url,
            "scheme": "",
            "netloc": "",
            "domain": "",
            "path": "",
            "query": ""
        }

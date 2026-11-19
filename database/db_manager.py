import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import config

def get_connection() -> sqlite3.Connection:
    """Creates and returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite schema if not already present."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Table: scans (Core metadata and aggregate scores)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                subject TEXT,
                sender TEXT,
                body_snippet TEXT,
                spam_score REAL,
                phishing_score REAL,
                url_max_score REAL,
                composite_score REAL,
                verdict TEXT,
                threat_level TEXT,
                num_urls INTEGER,
                summary_json TEXT
            )
        """)
        
        # Table: scan_urls (Per-URL details for extracted links)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                domain TEXT,
                risk_score REAL,
                is_suspicious INTEGER,
                flags_json TEXT,
                FOREIGN KEY (scan_id) REFERENCES scans (id) ON DELETE CASCADE
            )
        """)

        # Table: fraud_reports (Reports escalated to Cyber Crime / Helpline 1930)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                sender TEXT,
                subject TEXT,
                message_snippet TEXT,
                phishing_score REAL,
                composite_score REAL,
                reported_urls TEXT,
                status TEXT
            )
        """)
        
        conn.commit()

def save_scan(scan_data: Dict[str, Any], extracted_urls: List[Dict[str, Any]]) -> int:
    """
    Saves a full scan and all associated embedded URL analyses into SQLite.
    Returns the generated scan_id.
    """
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO scans (
                timestamp, subject, sender, body_snippet,
                spam_score, phishing_score, url_max_score, composite_score,
                verdict, threat_level, num_urls, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
            scan_data.get("subject", "No Subject"),
            scan_data.get("sender", "Unknown Sender"),
            scan_data.get("body_snippet", "")[:300],
            float(scan_data.get("spam_score", 0.0)),
            float(scan_data.get("phishing_score", 0.0)),
            float(scan_data.get("url_max_score", 0.0)),
            float(scan_data.get("composite_score", 0.0)),
            scan_data.get("verdict", "UNKNOWN"),
            scan_data.get("threat_level", "LOW"),
            len(extracted_urls),
            json.dumps(scan_data.get("summary", {}))
        ))
        
        scan_id = cursor.lastrowid
        
        for url_item in extracted_urls:
            cursor.execute("""
                INSERT INTO scan_urls (
                    scan_id, url, domain, risk_score, is_suspicious, flags_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                scan_id,
                url_item.get("url", ""),
                url_item.get("domain", ""),
                float(url_item.get("risk_score", 0.0)),
                1 if url_item.get("is_suspicious", False) else 0,
                json.dumps(url_item.get("reasons", []))
            ))
            
        conn.commit()
        return scan_id

def get_recent_scans(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves the most recent scans ordered by ID descending."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, subject, sender, spam_score, phishing_score,
                   url_max_score, composite_score, verdict, threat_level, num_urls
            FROM scans
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_scan_details(scan_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves full details for a scan including associated embedded URLs."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
        scan_row = cursor.fetchone()
        if not scan_row:
            return None
        
        scan_dict = dict(scan_row)
        if scan_dict.get("summary_json"):
            try:
                scan_dict["summary"] = json.loads(scan_dict["summary_json"])
            except Exception:
                scan_dict["summary"] = {}
                
        cursor.execute("SELECT * FROM scan_urls WHERE scan_id = ?", (scan_id,))
        url_rows = cursor.fetchall()
        urls_list = []
        for u in url_rows:
            u_dict = dict(u)
            if u_dict.get("flags_json"):
                try:
                    u_dict["reasons"] = json.loads(u_dict["flags_json"])
                except Exception:
                    u_dict["reasons"] = []
            urls_list.append(u_dict)
            
        scan_dict["urls"] = urls_list
        return scan_dict

def get_system_stats() -> Dict[str, Any]:
    """Computes aggregate analytics from stored scans."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total_scans FROM scans")
        total_scans = cursor.fetchone()["total_scans"]
        
        cursor.execute("SELECT COUNT(*) AS phishing_count FROM scans WHERE threat_level IN ('CRITICAL', 'HIGH')")
        phishing_count = cursor.fetchone()["phishing_count"]
        
        cursor.execute("SELECT COUNT(*) AS spam_count FROM scans WHERE verdict LIKE '%SPAM%'")
        spam_count = cursor.fetchone()["spam_count"]
        
        cursor.execute("SELECT COUNT(*) AS malicious_urls FROM scan_urls WHERE is_suspicious = 1")
        malicious_urls = cursor.fetchone()["malicious_urls"]

        cursor.execute("SELECT COUNT(*) AS total_reports FROM fraud_reports")
        total_reports = cursor.fetchone()["total_reports"]
        
        return {
            "total_scans": total_scans,
            "phishing_count": phishing_count,
            "spam_count": spam_count,
            "malicious_urls": malicious_urls,
            "total_reports": total_reports
        }

def save_fraud_report(report_data: Dict[str, Any]) -> str:
    """
    Saves an escalated fraud report into SQLite with an incident docket ID.
    Returns the generated report_id (e.g. CYBER-1930-XXXX).
    """
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        report_id = f"CYBER-1930-{int(datetime.utcnow().timestamp())}"
        
        cursor.execute("""
            INSERT INTO fraud_reports (
                report_id, timestamp, sender, subject, message_snippet,
                phishing_score, composite_score, reported_urls, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            report_data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
            report_data.get("sender", "Unknown"),
            report_data.get("subject", "No Subject"),
            report_data.get("message_snippet", "")[:500],
            float(report_data.get("phishing_score", 0.0)),
            float(report_data.get("composite_score", 0.0)),
            json.dumps(report_data.get("urls", [])),
            report_data.get("status", "SUBMITTED_TO_LOCAL_AUDIT")
        ))
        conn.commit()
        return report_id

def get_fraud_reports(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves list of logged fraud incident reports."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, report_id, timestamp, sender, subject, phishing_score, composite_score, status
            FROM fraud_reports
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def clear_all_history():
    """Wipes the database tables for testing or resetting."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scan_urls")
        cursor.execute("DELETE FROM scans")
        conn.commit()

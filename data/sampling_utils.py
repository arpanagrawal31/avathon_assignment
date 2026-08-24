import email as email_mod
import hashlib
import re

def _body(msg):
    """Extract plain-text body from parsed email message."""
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/plain":
                pl = p.get_payload(decode=True)
                if pl:
                    return pl.decode("utf-8", errors="replace")
        return None
    pl = msg.get_payload(decode=True)
    return pl.decode("utf-8", errors="replace") if pl else msg.get_payload()


def clean_one(raw):
    """Full cleaning pipeline on one raw email.
    Returns dict with cleaned fields, or None if filtered out."""
    msg = email_mod.message_from_string(raw)
    body = _body(msg)
    if not body or not body.strip():
        return None

    subj = msg.get("Subject", "")

    # Normalise whitespace
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[^\S\n]+", " ", body).strip()

    # Length filter: 15–600 words (~20–780 tokens)
    wc = len(body.split())
    if not (15 <= wc <= 600):
        return None

    fp = hashlib.md5(re.sub(r"\s+", " ", body.lower().strip()).encode()).hexdigest()
    return {
        "message_id": msg.get("Message-ID", "").strip("<>"),
        "from":    msg.get("From", ""),
        "to":      msg.get("To", ""),
        "subject": subj,
        "date":    msg.get("Date", ""),
        "body":    body,
        "fingerprint": fp,
    }


def trigrams(text):
    """Word-level trigram set for Jaccard near-dedup."""
    w = text.lower().split()
    return {(w[i], w[i+1], w[i+2]) for i in range(len(w)-2)} if len(w) >= 3 else set()


def near_dup(tg, seen, thr=0.7):
    """Check if trigram set is a near-duplicate of any in seen list."""
    if not tg:
        return False
    for s in seen:
        if s and len(tg & s) / len(tg | s) >= thr:
            return True
    return False
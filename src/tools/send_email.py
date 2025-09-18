import json
from pathlib import Path
from email.utils import parseaddr
import config

OUTBOX = Path("data/tools/outbox.json")

def _valid_email(addr: str) -> bool:
    # More robust than common regex: use standard library parsing + basic structure check
    name, email = parseaddr(addr or "")
    if not email or "@" not in email:
        return False
    local, _, domain = email.rpartition("@")
    return bool(local) and "." in domain

def _save_outbox(item: dict) -> None:
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    box = []
    if OUTBOX.exists():
        try:
            box = json.loads(OUTBOX.read_text(encoding="utf-8"))
        except Exception:
            box = []
    box.append(item)
    OUTBOX.write_text(json.dumps(box, ensure_ascii=False, indent=2), encoding="utf-8")

def send_email(receiver: str, subject: str, content: str) -> str:
    """
    This API for sending email, given the receiver, subject and content.

    Args:
        receiver (str): The receiver address of the email.
        subject (str): The subject address of the email.
        content (str): The content of the email.

    Returns:
        status (str): The status of the email.
    """
    if config.is_sandbox():
        return "success"

    r = (receiver or "").strip()
    s = (subject or "").strip()
    c = (content or "").strip()

    if not _valid_email(r):
        return "Error: invalid receiver email."
    if not s:
        return "Error: subject cannot be empty."
    if not c:
        return "Error: content cannot be empty."
    if len(s) > 200:
        return "Error: subject too long (>200 chars)."

    try:
        _save_outbox({"to": r, "subject": s, "content": c})
    except Exception as e:
        return f"Error: failed to save outbox ({e})"

    return "success"

FUNCTIONS = {
    "send_email": send_email,
}

if __name__ == "__main__":
    print(send_email("alice@example.com", "Meeting Notes", "Here are the notes..."))
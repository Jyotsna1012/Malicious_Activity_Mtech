"""Optional SMTP email delivery; disabled unless SMTP settings are provided."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD_PERCENT", "20"))


def send_incident_alert(recipient: str, result: dict, snapshot_path: str | None) -> tuple[bool, str]:
    """Send one incident email when score exceeds threshold. Returns (sent, message)."""
    if result["threat_percent"] <= ALERT_THRESHOLD:
        return False, f"No alert: score is not above {ALERT_THRESHOLD:.0f}%."
    required = ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "ALERT_FROM")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        return False, "Alert threshold crossed, but SMTP is not configured (missing: " + ", ".join(missing) + ")."
    msg = EmailMessage()
    msg["Subject"] = f"Security incident screening alert ({result['threat_percent']:.1f}%)"
    msg["From"] = os.environ["ALERT_FROM"]
    msg["To"] = recipient
    msg.set_content(
        f"Automated screening alert\n\n{result['summary']}\n\n"
        f"Flagged-frame percentage: {result['threat_percent']:.1f}%\n"
        f"Peak detector confidence: {result['peak_confidence']:.1%}\n"
        f"Incident window: {result['incident_start'] or 'not detected'} - {result['incident_end'] or 'not detected'}\n\n"
        "This is an automated screening result. A human should review the evidence before taking action."
    )
    if snapshot_path and Path(snapshot_path).is_file():
        msg.add_attachment(Path(snapshot_path).read_bytes(), maintype="image", subtype="jpeg", filename="incident_snapshot.jpg")
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"]), timeout=20) as server:
            server.starttls()
            server.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
            server.send_message(msg)
        return True, "Email alert sent."
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        return False, f"Email alert could not be sent: {exc}"

"""SMTP alert dispatch with an explicit local-console fallback."""
from __future__ import annotations

import logging
import os
import smtplib
from html import escape
from email.message import EmailMessage
from pathlib import Path

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD_PERCENT", "20"))
LOGGER = logging.getLogger("activity_alerts")


def send_incident_alert(recipient: str, result: dict, evidence_path: str | None,
                        threshold: float | None = None) -> tuple[bool, str]:
    """Send an HTML alert above the configured percentage; log a local fallback.

    Gmail SMTP requires an App Password in ``SMTP_PASSWORD``. No credentials are
    printed. The attached storyboard is generated locally by the detector.
    """
    limit = ALERT_THRESHOLD if threshold is None else float(threshold)
    if float(result.get("threat_percent", 0.0)) <= limit:
        return False, f"No alert: activity score is not above {limit:.1f}%."
    required = ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "ALERT_FROM")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        LOGGER.warning("MOCK INCIDENT ALERT | to=%s | score=%.1f%% | %s | SMTP missing: %s",
                       recipient, result["threat_percent"], result["summary"], ", ".join(missing))
        return False, "Threshold crossed; SMTP is not configured. Alert written to the local application log."

    msg = EmailMessage()
    msg["Subject"] = f"Activity screening alert ({result['threat_percent']:.1f}%)"
    msg["From"] = os.environ["ALERT_FROM"]
    msg["To"] = recipient
    action_text = ", ".join(result.get("action_labels", {}).keys()) or "No target action label"
    object_text = ", ".join(result.get("object_counts", {}).keys()) or "None"
    plain = (
        "Automated local video screening alert\n\n"
        f"{result['summary']}\n\nDetected action labels: {action_text}\nDetected object labels: {object_text}\n"
        f"Flagged frames: {result.get('malicious_frames', 0)}/{result['frame_count']}\n"
        f"Threat percentage: {result['threat_percent']:.1f}%\n"
        f"Peak threat score: {result['peak_confidence']:.1%}\n"
        f"Incident windows: {result.get('incident_runs', [])}\n\n"
        "This is an automated screening cue. Review the original evidence before acting."
    )
    safe_summary = escape(str(result["summary"]))
    safe_actions = escape(action_text)
    safe_objects = escape(object_text)
    html = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033">
      <h2>Activity screening alert</h2><p>{safe_summary}</p>
      <ul><li>Action labels: {safe_actions}</li><li>Object labels: {safe_objects}</li>
      <li>Flagged frames: {result.get('malicious_frames', 0)}/{result['frame_count']}</li>
      <li>Threat percentage: {result['threat_percent']:.1f}% (alert threshold: {limit:.1f}%)</li>
      <li>Peak threat score: {result['peak_confidence']:.1%}</li></ul>
      <p>Automated screening cue only. A human should review the evidence.</p>
    </body></html>"""
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    if evidence_path and Path(evidence_path).is_file():
        path = Path(evidence_path)
        mime = "png" if path.suffix.lower() == ".png" else "jpeg"
        msg.add_attachment(path.read_bytes(), maintype="image", subtype=mime,
                           filename=f"incident_storyboard{path.suffix or '.jpg'}")
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"]), timeout=25) as server:
            server.starttls()
            server.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
            server.send_message(msg)
        return True, "Email alert sent with storyboard evidence."
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        LOGGER.exception("SMTP alert failed; falling back to local log")
        LOGGER.warning("MOCK INCIDENT ALERT | to=%s | score=%.1f%% | %s",
                       recipient, result["threat_percent"], result["summary"])
        return False, f"Email failed and alert was written to the local log ({type(exc).__name__})."

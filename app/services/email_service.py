"""
EmailService — IMAP inbox reader and SMTP sender.
"""

from __future__ import annotations

import email
import imaplib
import smtplib
from email import encoders
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from typing import Optional

from bs4 import BeautifulSoup

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _decode_mime_words(s: str) -> str:
    """Decode MIME-encoded header words (e.g. =?UTF-8?B?...?=)."""
    if not s:
        return ""
    decoded_parts = decode_header(s)
    parts = []
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            parts.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            parts.append(str(part))
    return "".join(parts)


class EmailService:
    def __init__(self) -> None:
        self._imap_conn: Optional[imaplib.IMAP4_SSL] = None

    # ------------------------------------------------------------------
    # IMAP connection
    # ------------------------------------------------------------------

    def connect_imap(self) -> imaplib.IMAP4_SSL:
        """Open an IMAP4 SSL connection and authenticate."""
        if self._imap_conn:
            try:
                self._imap_conn.noop()
                return self._imap_conn
            except Exception:
                self._imap_conn = None

        logger.info(f"Connecting to IMAP {settings.imap_host}:{settings.imap_port}")
        conn = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        conn.login(settings.imap_user, settings.imap_pass)
        self._imap_conn = conn
        return conn

    def _disconnect_imap(self) -> None:
        if self._imap_conn:
            try:
                self._imap_conn.logout()
            except Exception:
                pass
            self._imap_conn = None

    # ------------------------------------------------------------------
    # Inbox reading
    # ------------------------------------------------------------------

    def fetch_unread_emails(self) -> list[dict]:
        """
        Fetch all UNSEEN emails from INBOX.

        Returns list of dicts:
            {msg_id, subject, body, sender, html_body}
        Where sender is (name, email) tuple.
        """
        results: list[dict] = []

        try:
            conn = self.connect_imap()
            conn.select("INBOX")

            _, data = conn.search(None, "UNSEEN")
            msg_ids = data[0].split()
            logger.info(f"fetch_unread_emails: found {len(msg_ids)} unread messages")

            for msg_id_bytes in msg_ids:
                try:
                    _, msg_data = conn.fetch(msg_id_bytes, "(RFC822)")
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    msg_id_str = msg.get("Message-ID", msg_id_bytes.decode())
                    subject = _decode_mime_words(msg.get("Subject", ""))
                    sender = self.parse_sender(msg)
                    body, html_body = self._extract_body(msg)

                    results.append({
                        "msg_id": msg_id_str,
                        "imap_uid": msg_id_bytes.decode(),
                        "subject": subject,
                        "body": body,
                        "html_body": html_body,
                        "sender": sender,
                    })
                except Exception as exc:
                    logger.warning(f"Failed to parse email {msg_id_bytes}: {exc}")

        except Exception as exc:
            logger.error(f"fetch_unread_emails failed: {exc}")
            self._disconnect_imap()
            raise

        return results

    def parse_sender(self, msg: email.message.Message) -> tuple[str, str]:
        """Parse From header → (display_name, email_address)."""
        raw_from = msg.get("From", "")
        name, addr = parseaddr(raw_from)
        name = _decode_mime_words(name)
        return (name.strip(), addr.strip().lower())

    def _extract_body(self, msg: email.message.Message) -> tuple[str, str]:
        """Extract plain-text and HTML body from a MIME message."""
        plain_parts: list[str] = []
        html_parts: list[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in disposition:
                    continue
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                decoded = payload.decode(charset, errors="replace")
                if ct == "text/plain":
                    plain_parts.append(decoded)
                elif ct == "text/html":
                    html_parts.append(decoded)
        else:
            ct = msg.get_content_type()
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload:
                decoded = payload.decode(charset, errors="replace")
                if ct == "text/html":
                    html_parts.append(decoded)
                else:
                    plain_parts.append(decoded)

        html_body = "\n".join(html_parts)

        # Prefer plain text; if not available, strip HTML
        if plain_parts:
            plain_body = "\n".join(plain_parts)
        elif html_body:
            soup = BeautifulSoup(html_body, "html.parser")
            plain_body = soup.get_text(separator="\n", strip=True)
        else:
            plain_body = ""

        return plain_body[:50_000], html_body[:100_000]

    def mark_processed(self, msg_id: str) -> None:
        """Mark email as Seen by Message-ID search."""
        try:
            conn = self.connect_imap()
            conn.select("INBOX")
            _, data = conn.search(None, f'HEADER Message-ID "{msg_id}"')
            for uid in data[0].split():
                conn.store(uid, "+FLAGS", "\\Seen")
            logger.debug(f"Marked email {msg_id} as Seen")
        except Exception as exc:
            logger.warning(f"mark_processed failed for {msg_id}: {exc}")

    # ------------------------------------------------------------------
    # SMTP sending
    # ------------------------------------------------------------------

    def send_acknowledgment(
        self,
        to_email: str,
        ticket_id: int,
        category: str,
        estimated_response_hours: int,
    ) -> None:
        """Send HTML acknowledgment email to customer."""
        category_display = category.replace("_", " ").title()

        subject = f"[Ticket #{ticket_id}] We've received your {category_display} request"
        html_body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: auto; padding: 20px;">
  <div style="background: #0066cc; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
    <h2 style="margin: 0;">Support Ticket Received</h2>
  </div>
  <div style="background: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px;">
    <p>Thank you for contacting us. We have received your <strong>{category_display}</strong> request.</p>
    <table style="border-collapse: collapse; width: 100%; margin: 15px 0;">
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold; width: 40%;">Ticket Number</td>
        <td style="padding: 8px; border: 1px solid #ddd;"># {ticket_id}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">Category</td>
        <td style="padding: 8px; border: 1px solid #ddd;">{category_display}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">Expected Response</td>
        <td style="padding: 8px; border: 1px solid #ddd;">Within {estimated_response_hours} hour(s)</td>
      </tr>
    </table>
    <p>Our team will review your request and respond as soon as possible.</p>
    <p>Please keep your ticket number for reference.</p>
    <hr style="border: none; border-top: 1px solid #ddd;">
    <p style="font-size: 12px; color: #888;">This is an automated acknowledgment. Please do not reply to this email directly.</p>
  </div>
</body>
</html>
"""
        self._send_email(to_email, subject, html_body)

    def send_escalation_alert(self, escalation_email: str, ticket_data: dict) -> None:
        """Send escalation alert email to the escalation address."""
        ticket_id = ticket_data.get("id", "N/A")
        subject_line = ticket_data.get("subject", "")[:100]
        sender = ticket_data.get("sender_email", "")
        category = ticket_data.get("category", "").replace("_", " ").title()
        priority = ticket_data.get("priority", "").upper()
        sla_deadline = ticket_data.get("sla_deadline", "N/A")
        crm_url = ticket_data.get("crm_url", "")

        subject = f"[ESCALATION] Ticket #{ticket_id} SLA Breach — {priority} Priority"
        html_body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: auto; padding: 20px;">
  <div style="background: #cc0000; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
    <h2 style="margin: 0;">SLA Breach — Immediate Attention Required</h2>
  </div>
  <div style="background: #fff8f8; padding: 20px; border: 1px solid #ffcccc; border-top: none; border-radius: 0 0 5px 5px;">
    <p><strong>A ticket has breached its SLA deadline and requires immediate attention.</strong></p>
    <table style="border-collapse: collapse; width: 100%; margin: 15px 0;">
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold; width: 40%;">Ticket ID</td>
        <td style="padding: 8px; border: 1px solid #ddd;">#{ticket_id}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">Subject</td>
        <td style="padding: 8px; border: 1px solid #ddd;">{subject_line}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">Customer</td>
        <td style="padding: 8px; border: 1px solid #ddd;">{sender}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">Category</td>
        <td style="padding: 8px; border: 1px solid #ddd;">{category}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">Priority</td>
        <td style="padding: 8px; border: 1px solid #ddd; color: #cc0000; font-weight: bold;">{priority}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;">SLA Deadline</td>
        <td style="padding: 8px; border: 1px solid #ddd;">{sla_deadline}</td>
      </tr>
      {"<tr><td style='padding: 8px; border: 1px solid #ddd; background: #eee; font-weight: bold;'>CRM Ticket</td><td style='padding: 8px; border: 1px solid #ddd;'><a href='" + crm_url + "'>View in CRM</a></td></tr>" if crm_url else ""}
    </table>
    <p style="color: #cc0000; font-weight: bold;">Please take immediate action to resolve this ticket.</p>
  </div>
</body>
</html>
"""
        self._send_email(escalation_email, subject, html_body)

    def _send_email(self, to_email: str, subject: str, html_body: str) -> None:
        """Send an HTML email via SMTP."""
        if not settings.smtp_user:
            logger.warning("SMTP not configured — skipping email send")
            return

        msg = MIMEMultipart("alternative")
        msg["From"] = settings.smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.ehlo()
                server.starttls()
                if settings.smtp_user and settings.smtp_pass:
                    server.login(settings.smtp_user, settings.smtp_pass)
                server.sendmail(settings.smtp_user, to_email, msg.as_string())
            logger.info(f"Email sent to {to_email}: {subject}")
        except Exception as exc:
            logger.error(f"SMTP send failed to {to_email}: {exc}")
            raise

"""
Celery tasks for support ticket automation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.tasks.celery_app import celery_app
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Sync DB helper (psycopg2 for Celery workers)
# ---------------------------------------------------------------------------

def _get_sync_db_url() -> str:
    return settings.database_url.replace(
        "postgresql+asyncpg://", "postgresql+psycopg2://"
    )


_sync_engine = None
_SyncSession: Optional[sessionmaker] = None


def _get_session() -> Session:
    global _sync_engine, _SyncSession
    if _sync_engine is None:
        _sync_engine = create_engine(
            _get_sync_db_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        _SyncSession = sessionmaker(bind=_sync_engine, autoflush=False, autocommit=False)
    return _SyncSession()


# ---------------------------------------------------------------------------
# Task: process_ticket_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.tasks.ticket_tasks.process_ticket_task",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue="tickets",
)
def process_ticket_task(self, ticket_id: int) -> dict:
    """classify → set priority → set SLA → create CRM ticket → send ack email."""
    from app.models import Ticket, TicketLog, CategoryEnum, PriorityEnum, SentimentEnum, TicketStatusEnum
    from app.services.classifier_service import TicketClassifier
    from app.automation.crm_bot import CRMBot

    logger.info(f"process_ticket_task: ticket_id={ticket_id}")
    session = _get_session()

    try:
        ticket = session.get(Ticket, ticket_id)
        if not ticket:
            logger.error(f"Ticket {ticket_id} not found")
            return {"error": "ticket not found"}

        # 1. Classify
        classifier = TicketClassifier()
        classifier.load_or_train()
        result = classifier.classify(ticket.subject, ticket.body)

        ticket.category = CategoryEnum(result.category)
        ticket.sentiment = SentimentEnum(result.sentiment)
        ticket.status = TicketStatusEnum.categorized

        # 2. Set priority based on sentiment + category
        priority = _determine_priority(result.category, result.sentiment)
        ticket.priority = PriorityEnum(priority)

        # 3. Set SLA deadline
        ticket.sla_deadline = _calculate_sla(priority)

        session.add(TicketLog(
            ticket_id=ticket_id,
            action="classified",
            details={
                "category": result.category,
                "sentiment": result.sentiment,
                "priority": priority,
                "category_confidence": result.category_confidence,
                "sentiment_score": result.sentiment_score,
            },
        ))
        session.commit()

        # 4. Create CRM ticket via Playwright
        crm_result = asyncio.run(_create_crm_ticket_async(ticket))

        if crm_result.get("crm_ticket_id"):
            ticket.crm_ticket_id = crm_result["crm_ticket_id"]
            ticket.crm_url = crm_result.get("url", "")
            ticket.status = TicketStatusEnum.crm_created
            session.add(TicketLog(
                ticket_id=ticket_id,
                action="crm_created",
                details=crm_result,
            ))
        else:
            ticket.status = TicketStatusEnum.failed
            ticket.error_message = crm_result.get("error", "CRM creation failed")
            session.add(TicketLog(
                ticket_id=ticket_id,
                action="crm_failed",
                details=crm_result,
            ))

        session.commit()

        # 5. Send acknowledgment email
        send_acknowledgment_task.apply_async(args=[ticket_id], queue="notifications")

        return {
            "ticket_id": ticket_id,
            "category": result.category,
            "priority": priority,
            "crm_ticket_id": ticket.crm_ticket_id,
        }

    except Exception as exc:
        session.rollback()
        logger.exception(f"process_ticket_task failed for ticket_id={ticket_id}: {exc}")

        # Update ticket error state
        try:
            ticket = session.get(Ticket, ticket_id)
            if ticket:
                from app.models import TicketStatusEnum
                ticket.status = TicketStatusEnum.failed
                ticket.error_message = str(exc)
                ticket.retry_count = (ticket.retry_count or 0) + 1
                session.commit()
        except Exception:
            pass

        raise self.retry(exc=exc)

    finally:
        session.close()


async def _create_crm_ticket_async(ticket) -> dict:
    """Bridge for running async CRMBot from sync Celery context."""
    from app.automation.crm_bot import CRMBot

    ticket_data = {
        "id": ticket.id,
        "subject": ticket.subject,
        "body": ticket.body,
        "sender_email": ticket.sender_email,
        "sender_name": ticket.sender_name or "",
        "category": ticket.category.value if hasattr(ticket.category, "value") else str(ticket.category),
        "priority": ticket.priority.value if hasattr(ticket.priority, "value") else str(ticket.priority),
        "sentiment": ticket.sentiment.value if hasattr(ticket.sentiment, "value") else str(ticket.sentiment),
    }

    async with CRMBot() as bot:
        page = await bot._context.new_page()
        try:
            await bot.login(page)
            return await bot.create_support_ticket(page, ticket_data)
        finally:
            await page.close()


def _determine_priority(category: str, sentiment: str) -> str:
    """Derive priority from category and sentiment."""
    # Complaint + angry = critical
    if category == "complaint" and sentiment == "angry":
        return "critical"
    # Any angry = high
    if sentiment == "angry":
        return "high"
    # Refund or complaint = high
    if category in ("refund", "complaint"):
        return "high"
    # Billing + negative = high
    if category == "billing" and sentiment == "negative":
        return "high"
    # Technical issues = normal
    if category == "technical":
        return "normal"
    # Everything else = normal or low
    if sentiment == "positive":
        return "low"
    return "normal"


def _calculate_sla(priority: str) -> datetime:
    """Calculate SLA deadline based on priority."""
    now = datetime.now(timezone.utc)
    hours = {
        "critical": settings.sla_critical_hours,
        "high": settings.sla_high_hours,
        "normal": settings.sla_normal_hours,
        "low": settings.sla_normal_hours * 2,
    }.get(priority, settings.sla_normal_hours)
    return now + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Task: scan_inbox_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.tasks.ticket_tasks.scan_inbox_task",
    queue="inbox",
)
def scan_inbox_task() -> dict:
    """Read IMAP inbox, create Ticket records, dispatch process_ticket_task."""
    from app.models import Ticket, TicketLog, TicketStatusEnum
    from app.services.email_service import EmailService

    logger.info("scan_inbox_task: scanning inbox")
    email_svc = EmailService()
    session = _get_session()
    new_tickets = 0

    try:
        emails = email_svc.fetch_unread_emails()
        logger.info(f"scan_inbox_task: found {len(emails)} unread emails")

        for email_data in emails:
            msg_id = email_data.get("msg_id", "")

            # Skip if already processed
            existing = session.execute(
                select(Ticket).where(Ticket.email_id == msg_id)
            ).scalar_one_or_none()
            if existing:
                continue

            sender_name, sender_email = email_data.get("sender", ("", ""))
            subject = email_data.get("subject", "(no subject)")
            body = email_data.get("body", "")

            ticket = Ticket(
                email_id=msg_id,
                subject=subject[:1000],
                body=body,
                sender_email=sender_email[:255],
                sender_name=sender_name[:255] if sender_name else None,
                status=TicketStatusEnum.new,
            )
            session.add(ticket)
            session.flush()

            session.add(TicketLog(
                ticket_id=ticket.id,
                action="created_from_email",
                details={"msg_id": msg_id, "sender": sender_email},
            ))
            session.commit()

            # Mark email as processed
            try:
                email_svc.mark_processed(msg_id)
            except Exception as mark_exc:
                logger.warning(f"Could not mark email {msg_id} as processed: {mark_exc}")

            # Dispatch processing task
            process_ticket_task.apply_async(args=[ticket.id], queue="tickets")
            new_tickets += 1

        logger.info(f"scan_inbox_task: created {new_tickets} new tickets")
        return {"new_tickets": new_tickets}

    except Exception as exc:
        session.rollback()
        logger.exception(f"scan_inbox_task failed: {exc}")
        raise

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Task: check_sla_breaches_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.tasks.ticket_tasks.check_sla_breaches_task",
    queue="sla",
)
def check_sla_breaches_task() -> dict:
    """Find tickets past SLA deadline, mark escalated, send escalation emails."""
    from app.models import Ticket, TicketLog, TicketStatusEnum
    from app.services.email_service import EmailService

    logger.info("check_sla_breaches_task: checking SLA breaches")
    now = datetime.now(timezone.utc)
    session = _get_session()
    email_svc = EmailService()
    escalated_count = 0

    try:
        breached = session.execute(
            select(Ticket).where(
                and_(
                    Ticket.sla_deadline <= now,
                    Ticket.is_escalated == False,  # noqa: E712
                    Ticket.status.notin_(
                        [TicketStatusEnum.closed, TicketStatusEnum.escalated]
                    ),
                )
            )
        ).scalars().all()

        for ticket in breached:
            ticket.is_escalated = True
            ticket.status = TicketStatusEnum.escalated

            session.add(TicketLog(
                ticket_id=ticket.id,
                action="sla_breach_escalated",
                details={
                    "sla_deadline": ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                    "breached_at": now.isoformat(),
                },
            ))

            # Send escalation email
            if settings.escalation_email:
                try:
                    ticket_data = {
                        "id": ticket.id,
                        "subject": ticket.subject,
                        "sender_email": ticket.sender_email,
                        "sender_name": ticket.sender_name or "",
                        "category": ticket.category.value if hasattr(ticket.category, "value") else str(ticket.category),
                        "priority": ticket.priority.value if hasattr(ticket.priority, "value") else str(ticket.priority),
                        "sla_deadline": ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                        "crm_url": ticket.crm_url or "",
                    }
                    email_svc.send_escalation_alert(settings.escalation_email, ticket_data)
                except Exception as email_exc:
                    logger.warning(f"Escalation email failed for ticket {ticket.id}: {email_exc}")

            escalated_count += 1

        session.commit()
        logger.info(f"check_sla_breaches_task: escalated {escalated_count} tickets")
        return {"escalated": escalated_count}

    except Exception as exc:
        session.rollback()
        logger.exception(f"check_sla_breaches_task failed: {exc}")
        raise

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Task: retry_failed_tickets_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.tasks.ticket_tasks.retry_failed_tickets_task",
    queue="retry",
)
def retry_failed_tickets_task() -> dict:
    """Re-dispatch failed tickets with retry_count < 3."""
    from app.models import Ticket, TicketStatusEnum

    logger.info("retry_failed_tickets_task: finding failed tickets to retry")
    session = _get_session()
    retried_count = 0

    try:
        failed_tickets = session.execute(
            select(Ticket).where(
                and_(
                    Ticket.status == TicketStatusEnum.failed,
                    Ticket.retry_count < 3,
                )
            )
        ).scalars().all()

        for ticket in failed_tickets:
            logger.info(f"Retrying ticket_id={ticket.id} (attempt {ticket.retry_count + 1})")
            process_ticket_task.apply_async(args=[ticket.id], queue="tickets")
            retried_count += 1

        logger.info(f"retry_failed_tickets_task: dispatched {retried_count} retries")
        return {"retried": retried_count}

    except Exception as exc:
        logger.exception(f"retry_failed_tickets_task failed: {exc}")
        raise

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Task: send_acknowledgment_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.tasks.ticket_tasks.send_acknowledgment_task",
    queue="notifications",
)
def send_acknowledgment_task(ticket_id: int) -> dict:
    """Send templated acknowledgment email to customer."""
    from app.models import Ticket, TicketLog, TicketStatusEnum
    from app.services.email_service import EmailService

    logger.info(f"send_acknowledgment_task: ticket_id={ticket_id}")
    session = _get_session()
    email_svc = EmailService()

    try:
        ticket = session.get(Ticket, ticket_id)
        if not ticket:
            logger.error(f"Ticket {ticket_id} not found for acknowledgment")
            return {"error": "ticket not found"}

        # Determine estimated response time based on priority
        priority = ticket.priority.value if hasattr(ticket.priority, "value") else str(ticket.priority)
        estimated_hours = {
            "critical": settings.sla_critical_hours,
            "high": settings.sla_high_hours,
            "normal": settings.sla_normal_hours,
            "low": settings.sla_normal_hours * 2,
        }.get(priority, settings.sla_normal_hours)

        category = ticket.category.value if hasattr(ticket.category, "value") else str(ticket.category)

        email_svc.send_acknowledgment(
            to_email=ticket.sender_email,
            ticket_id=ticket_id,
            category=category,
            estimated_response_hours=estimated_hours,
        )

        ticket.status = TicketStatusEnum.acknowledged
        session.add(TicketLog(
            ticket_id=ticket_id,
            action="acknowledgment_sent",
            details={"to": ticket.sender_email, "estimated_hours": estimated_hours},
        ))
        session.commit()

        logger.info(f"Acknowledgment sent for ticket_id={ticket_id} to {ticket.sender_email}")
        return {"ticket_id": ticket_id, "sent_to": ticket.sender_email}

    except Exception as exc:
        session.rollback()
        logger.exception(f"send_acknowledgment_task failed for ticket_id={ticket_id}: {exc}")
        raise

    finally:
        session.close()

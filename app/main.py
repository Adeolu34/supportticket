"""
FastAPI application for Support Ticket Automation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, func, desc, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db, init_db, close_db
from app.models import (
    Ticket,
    TicketLog,
    CategoryEnum,
    PriorityEnum,
    SentimentEnum,
    TicketStatusEnum,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Support Ticket API")
    await init_db()
    yield
    logger.info("Shutting down Support Ticket API")
    await close_db()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Support Ticket Automation API",
    description="Automated email-to-CRM support ticket pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TicketResponse(BaseModel):
    id: int
    email_id: Optional[str]
    subject: str
    body: str
    sender_email: str
    sender_name: Optional[str]
    category: str
    priority: str
    sentiment: str
    status: str
    crm_ticket_id: Optional[str]
    crm_url: Optional[str]
    sla_deadline: Optional[datetime]
    is_escalated: bool
    retry_count: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TicketLogResponse(BaseModel):
    id: int
    ticket_id: int
    action: str
    details: Optional[dict]
    created_at: datetime

    class Config:
        from_attributes = True


class ManualTicketRequest(BaseModel):
    subject: str
    body: str
    sender_email: str
    sender_name: Optional[str] = None


class RetrainRequest(BaseModel):
    training_data: Optional[list[list[str]]] = None  # [[text, label], ...]


# ---------------------------------------------------------------------------
# Ticket endpoints
# ---------------------------------------------------------------------------

@app.get("/tickets", response_model=list[TicketResponse], summary="List tickets")
async def list_tickets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    is_escalated: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    q = select(Ticket).order_by(desc(Ticket.created_at))

    if status:
        try:
            q = q.where(Ticket.status == TicketStatusEnum(status))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if category:
        try:
            q = q.where(Ticket.category == CategoryEnum(category))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    if priority:
        try:
            q = q.where(Ticket.priority == PriorityEnum(priority))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

    if is_escalated is not None:
        q = q.where(Ticket.is_escalated == is_escalated)

    q = q.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(q)
    tickets = result.scalars().all()

    return [
        TicketResponse(
            id=t.id,
            email_id=t.email_id,
            subject=t.subject,
            body=t.body,
            sender_email=t.sender_email,
            sender_name=t.sender_name,
            category=t.category.value,
            priority=t.priority.value,
            sentiment=t.sentiment.value,
            status=t.status.value,
            crm_ticket_id=t.crm_ticket_id,
            crm_url=t.crm_url,
            sla_deadline=t.sla_deadline,
            is_escalated=t.is_escalated,
            retry_count=t.retry_count,
            error_message=t.error_message,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in tickets
    ]


@app.get("/tickets/stats", summary="Ticket statistics")
async def ticket_stats(db: AsyncSession = Depends(get_db)):
    """Stats: by category, priority, SLA compliance rate."""
    now = datetime.now(timezone.utc)

    # By category
    cat_q = await db.execute(
        select(Ticket.category, func.count(Ticket.id).label("count"))
        .group_by(Ticket.category)
    )
    by_category = {row.category.value: row.count for row in cat_q.all()}

    # By priority
    pri_q = await db.execute(
        select(Ticket.priority, func.count(Ticket.id).label("count"))
        .group_by(Ticket.priority)
    )
    by_priority = {row.priority.value: row.count for row in pri_q.all()}

    # By status
    stat_q = await db.execute(
        select(Ticket.status, func.count(Ticket.id).label("count"))
        .group_by(Ticket.status)
    )
    by_status = {row.status.value: row.count for row in stat_q.all()}

    # SLA compliance: tickets with sla_deadline in the past
    total_with_sla = (await db.execute(
        select(func.count(Ticket.id)).where(Ticket.sla_deadline != None)  # noqa: E711
    )).scalar_one()

    breached = (await db.execute(
        select(func.count(Ticket.id)).where(
            and_(
                Ticket.sla_deadline <= now,
                Ticket.status.notin_([TicketStatusEnum.closed]),
            )
        )
    )).scalar_one()

    sla_compliance_rate = (
        round((total_with_sla - breached) / total_with_sla * 100, 1)
        if total_with_sla > 0
        else 100.0
    )

    total = (await db.execute(select(func.count(Ticket.id)))).scalar_one()
    escalated = (await db.execute(
        select(func.count(Ticket.id)).where(Ticket.is_escalated == True)  # noqa: E712
    )).scalar_one()

    return {
        "total_tickets": total,
        "escalated_tickets": escalated,
        "sla_compliance_rate_percent": sla_compliance_rate,
        "sla_breached_open": breached,
        "by_category": by_category,
        "by_priority": by_priority,
        "by_status": by_status,
    }


@app.get("/tickets/{ticket_id}", response_model=TicketResponse, summary="Get ticket by ID")
async def get_ticket(ticket_id: int, db: AsyncSession = Depends(get_db)):
    ticket = await db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketResponse(
        id=ticket.id,
        email_id=ticket.email_id,
        subject=ticket.subject,
        body=ticket.body,
        sender_email=ticket.sender_email,
        sender_name=ticket.sender_name,
        category=ticket.category.value,
        priority=ticket.priority.value,
        sentiment=ticket.sentiment.value,
        status=ticket.status.value,
        crm_ticket_id=ticket.crm_ticket_id,
        crm_url=ticket.crm_url,
        sla_deadline=ticket.sla_deadline,
        is_escalated=ticket.is_escalated,
        retry_count=ticket.retry_count,
        error_message=ticket.error_message,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
    )


@app.post("/tickets", response_model=TicketResponse, status_code=201, summary="Manual ticket submission")
async def create_ticket(
    body: ManualTicketRequest,
    db: AsyncSession = Depends(get_db),
):
    """Manually submit a support ticket (bypasses IMAP, goes straight to processing)."""
    from app.tasks.ticket_tasks import process_ticket_task

    ticket = Ticket(
        subject=body.subject[:1000],
        body=body.body,
        sender_email=body.sender_email,
        sender_name=body.sender_name,
        status=TicketStatusEnum.new,
    )
    db.add(ticket)
    await db.flush()
    await db.refresh(ticket)

    log = TicketLog(
        ticket_id=ticket.id,
        action="manual_submission",
        details={"sender": body.sender_email},
    )
    db.add(log)
    await db.commit()
    await db.refresh(ticket)

    # Dispatch processing
    process_ticket_task.apply_async(args=[ticket.id], queue="tickets")

    return TicketResponse(
        id=ticket.id,
        email_id=ticket.email_id,
        subject=ticket.subject,
        body=ticket.body,
        sender_email=ticket.sender_email,
        sender_name=ticket.sender_name,
        category=ticket.category.value,
        priority=ticket.priority.value,
        sentiment=ticket.sentiment.value,
        status=ticket.status.value,
        crm_ticket_id=ticket.crm_ticket_id,
        crm_url=ticket.crm_url,
        sla_deadline=ticket.sla_deadline,
        is_escalated=ticket.is_escalated,
        retry_count=ticket.retry_count,
        error_message=ticket.error_message,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
    )


@app.post("/tickets/{ticket_id}/retry", summary="Retry failed ticket")
async def retry_ticket(ticket_id: int, db: AsyncSession = Depends(get_db)):
    from app.tasks.ticket_tasks import process_ticket_task

    ticket = await db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if ticket.retry_count >= 3:
        raise HTTPException(
            status_code=400,
            detail=f"Ticket {ticket_id} has exceeded max retries (3)"
        )

    ticket.status = TicketStatusEnum.new
    ticket.error_message = None
    db.add(TicketLog(
        ticket_id=ticket_id,
        action="manual_retry",
        details={"triggered_at": datetime.now(timezone.utc).isoformat()},
    ))
    await db.commit()

    process_ticket_task.apply_async(args=[ticket_id], queue="tickets")
    return {"ticket_id": ticket_id, "message": "Retry dispatched"}


@app.post("/tickets/{ticket_id}/escalate", summary="Manually escalate a ticket")
async def escalate_ticket(ticket_id: int, db: AsyncSession = Depends(get_db)):
    from app.services.email_service import EmailService

    ticket = await db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.is_escalated = True
    ticket.status = TicketStatusEnum.escalated
    db.add(TicketLog(
        ticket_id=ticket_id,
        action="manually_escalated",
        details={"escalated_at": datetime.now(timezone.utc).isoformat()},
    ))
    await db.commit()

    # Send escalation email
    if settings.escalation_email:
        try:
            email_svc = EmailService()
            ticket_data = {
                "id": ticket.id,
                "subject": ticket.subject,
                "sender_email": ticket.sender_email,
                "sender_name": ticket.sender_name or "",
                "category": ticket.category.value,
                "priority": ticket.priority.value,
                "sla_deadline": ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                "crm_url": ticket.crm_url or "",
            }
            email_svc.send_escalation_alert(settings.escalation_email, ticket_data)
        except Exception as exc:
            logger.warning(f"Escalation email failed for ticket {ticket_id}: {exc}")

    return {"ticket_id": ticket_id, "escalated": True}


# ---------------------------------------------------------------------------
# Classifier endpoint
# ---------------------------------------------------------------------------

@app.post("/classifier/retrain", summary="Trigger classifier retraining")
async def retrain_classifier(request: RetrainRequest):
    """
    Trigger retraining of the ticket classifier.

    Optionally supply training_data as [[text, label], ...] to augment seed data.
    """
    from app.services.classifier_service import TicketClassifier

    try:
        classifier = TicketClassifier()
        training_data = None
        if request.training_data:
            training_data = [(row[0], row[1]) for row in request.training_data if len(row) >= 2]

        classifier.train(training_data=training_data)
        return {
            "status": "retrained",
            "model_path": str(classifier._model_path),
            "extra_examples": len(training_data) if training_data else 0,
        }
    except Exception as exc:
        logger.exception(f"Classifier retraining failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CategoryEnum(str, enum.Enum):
    billing = "billing"
    technical = "technical"
    general = "general"
    complaint = "complaint"
    refund = "refund"
    unknown = "unknown"


class PriorityEnum(str, enum.Enum):
    critical = "critical"
    high = "high"
    normal = "normal"
    low = "low"


class SentimentEnum(str, enum.Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    angry = "angry"


class TicketStatusEnum(str, enum.Enum):
    new = "new"
    categorized = "categorized"
    crm_created = "crm_created"
    acknowledged = "acknowledged"
    escalated = "escalated"
    closed = "closed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Email metadata
    email_id: Mapped[Optional[str]] = mapped_column(String(500), unique=True, nullable=True)
    subject: Mapped[str] = mapped_column(String(1000), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    sender_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    sender_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Classification
    category: Mapped[CategoryEnum] = mapped_column(
        Enum(CategoryEnum), default=CategoryEnum.unknown, nullable=False
    )
    priority: Mapped[PriorityEnum] = mapped_column(
        Enum(PriorityEnum), default=PriorityEnum.normal, nullable=False
    )
    sentiment: Mapped[SentimentEnum] = mapped_column(
        Enum(SentimentEnum), default=SentimentEnum.neutral, nullable=False
    )

    # Lifecycle
    status: Mapped[TicketStatusEnum] = mapped_column(
        Enum(TicketStatusEnum), default=TicketStatusEnum.new, nullable=False
    )
    crm_ticket_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    crm_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    sla_deadline: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_escalated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    logs: Mapped[list["TicketLog"]] = relationship(
        "TicketLog", back_populates="ticket", cascade="all, delete-orphan"
    )


class TicketLog(Base):
    __tablename__ = "ticket_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="logs")

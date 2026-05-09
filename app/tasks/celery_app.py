from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "support_ticket",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.ticket_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
)

celery_app.conf.beat_schedule = {
    # Scan inbox every 5 minutes
    "scan-inbox": {
        "task": "app.tasks.ticket_tasks.scan_inbox_task",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "inbox"},
    },
    # Check SLA breaches every 15 minutes
    "check-sla-breaches": {
        "task": "app.tasks.ticket_tasks.check_sla_breaches_task",
        "schedule": crontab(minute="*/15"),
        "options": {"queue": "sla"},
    },
    # Retry failed tickets every 20 minutes
    "retry-failed-tickets": {
        "task": "app.tasks.ticket_tasks.retry_failed_tickets_task",
        "schedule": crontab(minute="*/20"),
        "options": {"queue": "retry"},
    },
}

celery_app.conf.task_queues = {
    "inbox": {},
    "tickets": {},
    "sla": {},
    "retry": {},
    "notifications": {},
    "default": {},
}
celery_app.conf.task_default_queue = "default"

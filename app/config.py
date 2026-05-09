from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/support_ticket"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # IMAP
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_user: str = ""
    imap_pass: str = ""

    # SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""

    # CRM
    crm_portal_url: str = "https://crm.example.com"
    crm_user: str = ""
    crm_pass: str = ""

    # Paths
    screenshots_dir: str = "/app/screenshots"

    # Logging
    log_level: str = "INFO"

    # SLA (hours)
    sla_critical_hours: int = 2
    sla_high_hours: int = 8
    sla_normal_hours: int = 24

    # Escalation
    escalation_email: str = ""

    # ML
    classifier_model_path: str = "/app/models/ticket_classifier.pkl"

    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url


@lru_cache()
def get_settings() -> Settings:
    return Settings()

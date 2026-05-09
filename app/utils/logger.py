import sys
from pathlib import Path
from loguru import logger as _loguru_logger

from app.config import get_settings

_configured = False


def _configure_logger() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    log_level = settings.log_level.upper()

    _loguru_logger.remove()

    # Console handler
    _loguru_logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=True,
    )

    # Rotating file handler
    log_dir = Path("/app/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    _loguru_logger.add(
        str(log_dir / "support_ticket_{time:YYYY-MM-DD}.log"),
        level=log_level,
        rotation="00:00",
        retention="30 days",
        compression="gz",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    _configured = True


def get_logger(name: str):
    """Return a loguru logger bound to *name*."""
    _configure_logger()
    return _loguru_logger.bind(name=name)

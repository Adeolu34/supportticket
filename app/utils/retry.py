"""
Retry decorators with exponential backoff.
Async variant: @async_retry(...)
Sync  variant: @sync_retry(...)
"""

import asyncio
import functools
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple, Type

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _screenshot_path(func_name: str) -> str:
    settings = get_settings()
    screenshots_dir = Path(settings.screenshots_dir)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return str(screenshots_dir / f"retry_failure_{func_name}_{ts}.png")


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Async retry with exponential backoff. Saves screenshot on final failure if `page` kwarg present."""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            page = kwargs.get("page")
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        if page is not None:
                            try:
                                path = _screenshot_path(func.__name__)
                                await page.screenshot(path=path, full_page=True)
                                logger.info(f"Failure screenshot saved: {path}")
                            except Exception as ss_exc:
                                logger.warning(f"Could not save screenshot: {ss_exc}")
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed "
                        f"({exc}). Retrying in {delay:.1f}s…"
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def sync_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Sync retry with exponential backoff. Saves screenshot on final failure if `page` kwarg present."""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            page = kwargs.get("page")
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        if page is not None:
                            try:
                                path = _screenshot_path(func.__name__)
                                page.screenshot(path=path, full_page=True)
                                logger.info(f"Failure screenshot saved: {path}")
                            except Exception as ss_exc:
                                logger.warning(f"Could not save screenshot: {ss_exc}")
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed "
                        f"({exc}). Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)

        return wrapper

    return decorator

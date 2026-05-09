"""
CRMBot — Playwright context manager for CRM portal automation.

Usage:
    async with CRMBot() as bot:
        page = await bot._context.new_page()
        await bot.login(page)
        result = await bot.create_support_ticket(page, ticket_data)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.retry import async_retry

logger = get_logger(__name__)
settings = get_settings()


class CRMBot:
    """Async context manager — manages Playwright browser lifecycle for CRM."""

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "CRMBot":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        await self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        logger.info("CRMBot: browser started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("CRMBot: browser closed")

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    @async_retry(max_attempts=3, base_delay=2.0)
    async def login(self, page: Page) -> None:
        """CRM portal login with retry."""
        login_url = f"{settings.crm_portal_url.rstrip('/')}/login"
        logger.info(f"CRMBot: logging in at {login_url}")

        await page.goto(login_url, wait_until="networkidle", timeout=30_000)

        # Fill username
        username_selectors = [
            'input[name="username"]',
            'input[name="email"]',
            'input[name="login"]',
            '#username',
            '#email',
        ]
        for sel in username_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.fill(settings.crm_user)
                    break
            except Exception:
                continue

        # Fill password
        password_selectors = [
            'input[name="password"]',
            '#password',
            'input[type="password"]',
        ]
        for sel in password_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.fill(settings.crm_pass)
                    break
            except Exception:
                continue

        # Submit
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button.login-btn',
            'button:has-text("Login")',
            'button:has-text("Sign In")',
        ]
        for sel in submit_selectors:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    await btn.click()
                    break
            except Exception:
                continue

        # Wait for redirect away from login page
        try:
            await page.wait_for_url(
                lambda url: "/login" not in url and "/signin" not in url,
                timeout=15_000,
            )
        except PlaywrightTimeoutError:
            # Some CRMs stay on the same URL after login — check for error messages
            error_selectors = ['.error', '.alert-danger', '.login-error', '#error-message']
            for sel in error_selectors:
                try:
                    err = page.locator(sel).first
                    if await err.count() > 0:
                        error_text = await err.inner_text()
                        raise RuntimeError(f"CRM login failed: {error_text}")
                except RuntimeError:
                    raise
                except Exception:
                    continue

        logger.info("CRMBot: login successful")

    # ------------------------------------------------------------------
    # Ticket creation
    # ------------------------------------------------------------------

    @async_retry(max_attempts=3, base_delay=2.0)
    async def create_support_ticket(self, page: Page, ticket: dict) -> dict:
        """
        Navigate CRM and fill ticket form.

        Returns:
            {crm_ticket_id: str, url: str, error: str|None}
        """
        try:
            from app.services.crm_service import CRMService
            crm_svc = CRMService()
            result = await crm_svc.create_ticket(ticket, page)

            # Determine correct department from category
            category = ticket.get("category", "general")
            department = self._category_to_department(category)

            if result.get("crm_ticket_id") and department:
                try:
                    await crm_svc.assign_department(page, result["crm_ticket_id"], department)
                except Exception as dept_exc:
                    logger.warning(f"Department assignment failed: {dept_exc}")

            return result

        except Exception as exc:
            logger.error(f"create_support_ticket failed: {exc}")
            await self._take_screenshot(page, f"crm_ticket_error_{ticket.get('id', 'unknown')}")
            return {"crm_ticket_id": None, "url": None, "error": str(exc)}

    # ------------------------------------------------------------------
    # Department assignment
    # ------------------------------------------------------------------

    @async_retry(max_attempts=2, base_delay=1.0)
    async def assign_to_department(
        self, page: Page, crm_ticket_id: str, department: str
    ) -> bool:
        """Navigate to ticket and assign to department."""
        try:
            from app.services.crm_service import CRMService
            crm_svc = CRMService()
            return await crm_svc.assign_department(page, crm_ticket_id, department)
        except Exception as exc:
            logger.error(f"assign_to_department failed for {crm_ticket_id}: {exc}")
            await self._take_screenshot(page, f"assign_dept_error_{crm_ticket_id}")
            return False

    # ------------------------------------------------------------------
    # Status update
    # ------------------------------------------------------------------

    @async_retry(max_attempts=3, base_delay=1.5)
    async def update_ticket_status(
        self, page: Page, crm_ticket_id: str, status: str
    ) -> bool:
        """Navigate to ticket and update its status field."""
        base_url = settings.crm_portal_url.rstrip("/")
        ticket_url = f"{base_url}/tickets/{crm_ticket_id}"

        try:
            await page.goto(ticket_url, wait_until="networkidle", timeout=20_000)

            status_selectors = [
                'select[name="status"]',
                '#ticket_status',
                'select[name="state"]',
            ]
            for sel in status_selectors:
                try:
                    loc = page.locator(sel).first
                    if await loc.count() > 0:
                        await loc.select_option(value=status, label=status.replace("_", " ").title())
                        # Save
                        save_btn = page.locator('button[type="submit"], button.save, button.btn-primary').first
                        if await save_btn.count() > 0:
                            await save_btn.click()
                            await page.wait_for_load_state("networkidle", timeout=15_000)
                        logger.info(f"CRMBot: updated ticket {crm_ticket_id} status to {status}")
                        return True
                except Exception:
                    continue

            logger.warning(f"CRMBot: could not update status for ticket {crm_ticket_id}")
            return False

        except Exception as exc:
            logger.error(f"update_ticket_status failed for {crm_ticket_id}: {exc}")
            await self._take_screenshot(page, f"status_update_error_{crm_ticket_id}")
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _take_screenshot(self, page: Page, label: str) -> None:
        """Save timestamped screenshot."""
        try:
            screenshots_dir = Path(settings.screenshots_dir)
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            path = str(screenshots_dir / f"{label}_{ts}.png")
            await page.screenshot(path=path, full_page=True)
            logger.info(f"Screenshot saved: {path}")
        except Exception as exc:
            logger.warning(f"Screenshot failed: {exc}")

    @staticmethod
    def _category_to_department(category: str) -> str:
        """Map ticket category to CRM department name."""
        mapping = {
            "billing": "Finance",
            "technical": "Technical Support",
            "complaint": "Customer Relations",
            "refund": "Finance",
            "general": "Customer Service",
            "unknown": "Customer Service",
        }
        return mapping.get(category, "Customer Service")

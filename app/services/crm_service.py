"""
CRMService — Playwright-based CRM portal interaction.

This class is designed to be used with an externally-managed Playwright
page (the CRMBot owns the browser context).
"""

from __future__ import annotations

import re
from typing import Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.retry import async_retry

logger = get_logger(__name__)
settings = get_settings()


class CRMService:
    """Playwright-based operations on the CRM portal."""

    def __init__(self) -> None:
        self.base_url = settings.crm_portal_url.rstrip("/")

    # ------------------------------------------------------------------
    # Ticket creation
    # ------------------------------------------------------------------

    @async_retry(max_attempts=3, base_delay=2.0)
    async def create_ticket(self, ticket_data: dict, page: Page) -> dict:
        """
        Log into CRM portal, fill ticket form, submit.

        Returns dict with crm_ticket_id and url.
        """
        new_ticket_url = f"{self.base_url}/tickets/new"
        logger.info(f"CRMService: navigating to {new_ticket_url}")

        await page.goto(new_ticket_url, wait_until="networkidle", timeout=30_000)

        # Fill title / subject
        title_selectors = [
            'input[name="title"]',
            'input[name="subject"]',
            '#ticket_title',
            '#ticket_subject',
            'input[placeholder*="title" i]',
            'input[placeholder*="subject" i]',
        ]
        for sel in title_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.fill(ticket_data.get("subject", "")[:500])
                    break
            except Exception:
                continue

        # Fill body / description
        body_selectors = [
            'textarea[name="body"]',
            'textarea[name="description"]',
            '#ticket_body',
            '#ticket_description',
            '.ticket-body',
        ]
        for sel in body_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.fill(ticket_data.get("body", "")[:5000])
                    break
            except Exception:
                continue

        # Select category
        category = ticket_data.get("category", "general")
        category_selectors = [
            'select[name="category"]',
            '#ticket_category',
            'select[name="type"]',
        ]
        for sel in category_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.select_option(value=category, label=category.replace("_", " ").title())
                    break
            except Exception:
                continue

        # Select priority
        priority = ticket_data.get("priority", "normal")
        priority_selectors = [
            'select[name="priority"]',
            '#ticket_priority',
        ]
        for sel in priority_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.select_option(value=priority, label=priority.title())
                    break
            except Exception:
                continue

        # Fill customer email
        email_selectors = [
            'input[name="customer_email"]',
            'input[name="requester_email"]',
            'input[name="email"]',
            '#customer_email',
        ]
        for sel in email_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.fill(ticket_data.get("sender_email", ""))
                    break
            except Exception:
                continue

        # Fill customer name
        name_selectors = [
            'input[name="customer_name"]',
            'input[name="requester_name"]',
            '#customer_name',
        ]
        for sel in name_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.fill(ticket_data.get("sender_name", ""))
                    break
            except Exception:
                continue

        # Submit form
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button.btn-primary',
            '#submit-ticket',
            'button:has-text("Create")',
            'button:has-text("Submit")',
            'button:has-text("Save")',
        ]
        for sel in submit_selectors:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    await btn.click()
                    await page.wait_for_load_state("networkidle", timeout=30_000)
                    break
            except Exception:
                continue

        # Extract CRM ticket ID from current URL or page content
        crm_ticket_id = self._extract_ticket_id(page.url)
        crm_url = page.url

        if not crm_ticket_id:
            crm_ticket_id = await self._extract_ticket_id_from_page(page)

        logger.info(f"CRMService: created ticket crm_id={crm_ticket_id}, url={crm_url}")
        return {"crm_ticket_id": crm_ticket_id, "url": crm_url}

    async def get_ticket_url(self, page: Page, crm_ticket_id: str) -> str:
        """Navigate to ticket and return its canonical URL."""
        ticket_url = f"{self.base_url}/tickets/{crm_ticket_id}"
        try:
            await page.goto(ticket_url, wait_until="domcontentloaded", timeout=20_000)
            return page.url
        except Exception as exc:
            logger.warning(f"get_ticket_url failed for {crm_ticket_id}: {exc}")
            return ticket_url

    @async_retry(max_attempts=3, base_delay=1.5)
    async def assign_department(self, page: Page, crm_ticket_id: str, department: str) -> bool:
        """Navigate to ticket and assign it to the correct department."""
        ticket_url = f"{self.base_url}/tickets/{crm_ticket_id}"
        await page.goto(ticket_url, wait_until="networkidle", timeout=20_000)

        dept_selectors = [
            'select[name="department"]',
            'select[name="team"]',
            '#ticket_department',
            '#assigned_team',
        ]
        for sel in dept_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.select_option(label=department, value=department.lower())
                    # Save
                    save_btn = page.locator('button[type="submit"], button.save').first
                    if await save_btn.count() > 0:
                        await save_btn.click()
                        await page.wait_for_load_state("networkidle", timeout=15_000)
                    logger.info(f"Assigned ticket {crm_ticket_id} to department: {department}")
                    return True
            except Exception:
                continue

        logger.warning(f"Could not assign department for ticket {crm_ticket_id}")
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ticket_id(url: str) -> Optional[str]:
        """Extract numeric or alphanumeric ticket ID from URL patterns like /tickets/123."""
        patterns = [
            r"/tickets/(\w+)",
            r"[?&]ticket_id=(\w+)",
            r"[?&]id=(\d+)",
            r"/t/(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    async def _extract_ticket_id_from_page(page: Page) -> Optional[str]:
        """Try to find ticket ID in page content when URL doesn't contain it."""
        selectors = [
            '[data-ticket-id]',
            '#ticket-id',
            '.ticket-id',
            'span.id',
        ]
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    val = await loc.get_attribute("data-ticket-id") or await loc.inner_text()
                    digits = re.sub(r"\D", "", val)
                    if digits:
                        return digits
            except Exception:
                continue
        return None

"""
Realistic Playwright browser setup for E2E tests against our own platform.

Intended targets: local dev, staging, and preview environments we control.
Pair this with a test-traffic allowlist on the platform side (see
E2E_BYPASS_HEADER below) rather than relying on fingerprint tweaks alone.

Usage:
    pip install playwright pytest   # not pytest-playwright: fixture names clash
    playwright install chromium
    E2E_BASE_URL=https://staging.example.com pytest tests/e2e
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from playwright.sync_api import Browser, BrowserContext, Locator, Page, Playwright, sync_playwright


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_URL = os.environ.get("E2E_BASE_URL", "https://www.apple.com/shop/buy-iphone/iphone-duo/7.6-inch-display-256gb-night-sky-unlocked")
HEADLESS = os.environ.get("E2E_HEADLESS", "1") != "0"

# Preferred fix for WAF / bot-protection false positives in test envs:
# configure the platform to skip bot checks when this header carries a
# secret token. Keep the token in CI secrets, never in the repo.
E2E_BYPASS_HEADER = "X-E2E-Test-Token"
E2E_BYPASS_TOKEN = os.environ.get("E2E_BYPASS_TOKEN")


@dataclass(frozen=True)
class DeviceProfile:
    """A coherent set of browser properties. Mismatches (e.g. a Mac UA with a
    Windows platform, or a mobile UA on a 1920px viewport) cause more
    breakage than any single default does, so keep these together."""

    name: str
    user_agent: str
    viewport: dict
    device_scale_factor: float
    locale: str
    timezone_id: str
    is_mobile: bool = False
    has_touch: bool = False
    extra_headers: dict = field(default_factory=dict)


# Keep these in sync with the Chromium build Playwright ships; a UA claiming
# a different major version than the real engine is itself a mismatch.
PROFILES: dict[str, DeviceProfile] = {
    "windows_chrome": DeviceProfile(
        name="windows_chrome",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
        device_scale_factor=1,
        locale="en-US",
        timezone_id="America/New_York",
    ),
    "mac_chrome": DeviceProfile(
        name="mac_chrome",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1440, "height": 900},
        device_scale_factor=2,
        locale="en-US",
        timezone_id="America/Los_Angeles",
    ),
}


# --------------------------------------------------------------------------- #
# Browser / context initialization
# --------------------------------------------------------------------------- #

CHROMIUM_ARGS = [
    # Stops Blink from setting navigator.webdriver = true and from exposing
    # the "controlled by automated software" state to page scripts.
    "--disable-blink-features=AutomationControlled",
]

# Belt-and-braces: runs before any page script in every frame.
INIT_SCRIPT = """
(() => {
  // navigator.webdriver -> undefined, as in a normal user session
  Object.defineProperty(Navigator.prototype, 'webdriver', {
    get: () => undefined,
    configurable: true,
  });

  // Headless Chromium reports an empty language list in some builds
  if (!navigator.languages || navigator.languages.length === 0) {
    Object.defineProperty(Navigator.prototype, 'languages', {
      get: () => ['en-US', 'en'],
      configurable: true,
    });
  }
})();
"""


def launch_browser(pw: Playwright, headless: bool = HEADLESS) -> Browser:
    # channel="chromium" uses the "new" headless mode, which runs the full
    # browser rather than the stripped-down headless shell, so rendering,
    # fonts, codecs and feature detection behave like headed Chrome.
    return pw.chromium.launch(
        headless=headless,
        channel="chromium",
        args=CHROMIUM_ARGS,
        ignore_default_args=["--enable-automation"],
    )


def new_context(browser: Browser, profile: DeviceProfile = PROFILES["windows_chrome"]) -> BrowserContext:
    headers = {"Accept-Language": f"{profile.locale},{profile.locale.split('-')[0]};q=0.9"}
    headers.update(profile.extra_headers)
    if E2E_BYPASS_TOKEN:
        headers[E2E_BYPASS_HEADER] = E2E_BYPASS_TOKEN

    context = browser.new_context(
        base_url=BASE_URL,
        user_agent=profile.user_agent,
        viewport=profile.viewport,
        screen=profile.viewport,
        device_scale_factor=profile.device_scale_factor,
        is_mobile=profile.is_mobile,
        has_touch=profile.has_touch,
        locale=profile.locale,
        timezone_id=profile.timezone_id,
        color_scheme="light",
        extra_http_headers=headers,
    )
    context.add_init_script(INIT_SCRIPT)
    context.set_default_timeout(15_000)
    return context


@contextmanager
def realistic_page(profile_name: str = "windows_chrome") -> Iterator[Page]:
    """Standalone helper: `with realistic_page() as page: page.goto("/")`."""
    with sync_playwright() as pw:
        browser = launch_browser(pw)
        context = new_context(browser, PROFILES[profile_name])
        try:
            yield context.new_page()
        finally:
            context.close()
            browser.close()


# --------------------------------------------------------------------------- #
# Human-paced interaction helpers
# --------------------------------------------------------------------------- #

class Human:
    """Wraps a Page with human-like pacing.

    Delays are seeded so a failing run can be replayed exactly: set
    E2E_SEED to the value printed in the failing test's output. Use these
    for flows whose behavior depends on timing (debounced search, typeahead,
    rate limits, idle timers). For everything else, Playwright's auto-waiting
    is faster and less flaky than random sleeps.
    """

    def __init__(self, page: Page, seed: int | None = None, speed: float = 1.0):
        self.page = page
        self.seed = seed if seed is not None else int(os.environ.get("E2E_SEED", random.randrange(1_000_000)))
        self.rng = random.Random(self.seed)
        self.speed = speed  # <1.0 = faster, e.g. 0.25 in CI

    def pause(self, low: float = 0.3, high: float = 1.2) -> None:
        time.sleep(self.rng.uniform(low, high) * self.speed)

    def think(self) -> None:
        """Longer pause, as when a user reads a page before acting."""
        self.pause(0.8, 2.5)

    def move_to(self, target: Locator) -> None:
        box = target.bounding_box()
        if box is None:
            target.scroll_into_view_if_needed()
            box = target.bounding_box()
        if box is None:
            return
        # Aim somewhere inside the element, not dead-center
        x = box["x"] + box["width"] * self.rng.uniform(0.3, 0.7)
        y = box["y"] + box["height"] * self.rng.uniform(0.3, 0.7)
        self.page.mouse.move(x, y, steps=self.rng.randint(8, 25))

    def click(self, target: Locator | str) -> None:
        loc = self.page.locator(target) if isinstance(target, str) else target
        loc.wait_for(state="visible")
        self.move_to(loc)
        self.pause(0.08, 0.3)
        loc.click(delay=self.rng.randint(40, 120))  # mousedown -> mouseup gap
        self.pause()

    def type(self, target: Locator | str, text: str, clear: bool = True) -> None:
        loc = self.page.locator(target) if isinstance(target, str) else target
        self.click(loc)
        if clear:
            loc.fill("")
        for ch in text:
            loc.press_sequentially(ch, delay=0)
            delay = self.rng.uniform(0.05, 0.18)
            if ch in " .,@":
                delay += self.rng.uniform(0.05, 0.25)
            time.sleep(delay * self.speed)
        self.pause()

    def scroll(self, pixels: int = 600) -> None:
        remaining = pixels
        while remaining > 0:
            step = min(remaining, self.rng.randint(80, 200))
            self.page.mouse.wheel(0, step)
            remaining -= step
            time.sleep(self.rng.uniform(0.03, 0.1) * self.speed)
        self.pause()


# --------------------------------------------------------------------------- #
# pytest fixtures (import into conftest.py, or keep this file as conftest.py)
# --------------------------------------------------------------------------- #

try:
    import pytest

    @pytest.fixture(scope="session")
    def browser() -> Iterator[Browser]:
        with sync_playwright() as pw:
            b = launch_browser(pw)
            yield b
            b.close()

    @pytest.fixture(params=list(PROFILES))
    def page(request, browser: Browser) -> Iterator[Page]:
        context = new_context(browser, PROFILES[request.param])
        p = context.new_page()
        yield p
        context.close()

    @pytest.fixture
    def human(page: Page, request) -> Human:
        speed = float(os.environ.get("E2E_HUMAN_SPEED", "1.0"))
        h = Human(page, speed=speed)
        print(f"[{request.node.name}] E2E_SEED={h.seed}")
        return h

except ImportError:  # pytest not installed; standalone use still works
    pass


# --------------------------------------------------------------------------- #
# Smoke check: python tests/e2e/realistic_browser.py
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    with realistic_page() as page:
        page.goto("/")
        print("webdriver:", page.evaluate("navigator.webdriver"))
        print("userAgent:", page.evaluate("navigator.userAgent"))
        print("languages:", page.evaluate("navigator.languages"))
        print("timezone:", page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone"))

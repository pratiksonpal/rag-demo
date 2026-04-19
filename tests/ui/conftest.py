"""
Pytest fixtures for Streamlit UI tests.

Starts a Streamlit server on port 8502, launches a headless Chromium browser,
and provides a `page` fixture that navigates to the app before each test.
Screenshots are captured automatically on test failure.
"""

import sys
import time
import subprocess
from pathlib import Path

import pytest
import requests
from playwright.sync_api import sync_playwright

PORT = 8502
BASE_URL = f"http://localhost:{PORT}"
STARTUP_TIMEOUT_S = 60


@pytest.fixture(scope="session")
def streamlit_server():
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--server.runOnSave", "false",
            "--global.developmentMode", "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        try:
            if requests.get(BASE_URL, timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        pytest.exit("Streamlit server failed to start within timeout")
    yield BASE_URL
    proc.kill()


@pytest.fixture(scope="session")
def browser(streamlit_server):
    pw = sync_playwright().start()
    b = pw.chromium.launch(headless=True)
    yield b
    b.close()
    pw.stop()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture()
def page(browser, streamlit_server, request):
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = context.new_page()
    pg.goto(streamlit_server, wait_until="domcontentloaded", timeout=30000)
    # Wait for sidebar AND Tab 1 content (selectbox) to fully render
    pg.wait_for_selector('[data-testid="stSidebar"]', timeout=30000)
    pg.wait_for_selector('[data-testid="stSelectbox"]', timeout=30000)
    yield pg
    # Capture screenshot on failure
    rep = getattr(request.node, "rep_call", None)
    if rep and rep.failed:
        shot_dir = Path("tests/ui/screenshots")
        shot_dir.mkdir(exist_ok=True)
        safe_name = request.node.name.replace("/", "_").replace("\\", "_")
        pg.screenshot(path=str(shot_dir / f"{safe_name}.png"))
    context.close()

"""
scripts/capture.py - Automated screenshot tool using headless Chrome and CDP.
Captures high-resolution before/after screenshots of the Streamlit dashboard.
"""

import os
import sys
import time
import json
import base64
import urllib.request
import subprocess
import tempfile
import asyncio
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    r"C:\Users\mgarg\.gemini\antigravity-ide\brain\248869bc-fca1-4d0c-b98f-944211416ca8"
)


async def capture_page_screenshot(
    output_filename: str,
    page_radio_text: str = None,
    width: int = 1366,
    height: int = 768,
    wait_seconds: float = 3.0
):
    """Launches headless Chrome, navigates to Streamlit, selects page if needed, and saves PNG."""
    temp_profile = tempfile.mkdtemp()
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9444",
        f"--user-data-dir={temp_profile}",
        "--disable-gpu",
        "--no-first-run",
        f"--window-size={width},{height}",
        "http://localhost:8501"
    ])

    try:
        # Give Chrome a moment to initialize CDP port
        await asyncio.sleep(3.5)
        
        # Query CDP HTTP endpoint for open tabs
        with urllib.request.urlopen("http://localhost:9444/json") as resp:
            tabs = json.loads(resp.read().decode())
        
        page_tab = next(t for t in tabs if t.get("url", "").startswith("http://localhost:8501"))
        ws_url = page_tab["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url) as ws:
            # Wait for Streamlit WebSocket to connect and render initial DOM
            await asyncio.sleep(wait_seconds)

            # If a specific page radio label was requested, click it via JavaScript
            if page_radio_text:
                click_js = f"""
                (function() {{
                    const labels = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] [data-testid="stRadio"] label'));
                    const target = labels.find(l => l.innerText.includes("{page_radio_text}"));
                    if (target) {{
                        target.click();
                        return true;
                    }}
                    return false;
                }})()
                """
                await ws.send(json.dumps({
                    "id": 10,
                    "method": "Runtime.evaluate",
                    "params": {"expression": click_js}
                }))
                await ws.recv()
                # Wait for page to re-render after radio click
                await asyncio.sleep(2.5)

            # Capture screenshot
            await ws.send(json.dumps({
                "id": 20,
                "method": "Page.captureScreenshot",
                "params": {"format": "png"}
            }))
            reply = json.loads(await ws.recv())
            img_data = base64.b64decode(reply["result"]["data"])

            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            out_path = os.path.join(ARTIFACT_DIR, output_filename)
            with open(out_path, "wb") as f:
                f.write(img_data)
            print(f"Captured: {output_filename} ({len(img_data)} bytes)")
            return out_path
    finally:
        proc.terminate()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "caselens"
    if mode == "before":
        print("Capturing BEFORE screenshots...")
        asyncio.run(capture_page_screenshot("before_overview.png", None))
        asyncio.run(capture_page_screenshot("before_privacy.png", "Survivor Privacy"))
    elif mode in ("after", "caselens"):
        print("Capturing CASELENS screenshots...")
        asyncio.run(capture_page_screenshot("caselens_overview.png", None))
        asyncio.run(capture_page_screenshot("caselens_privacy.png", "Survivor Privacy"))
    elif mode == "all_7":
        print("Capturing all 7 pages...")
        pages = [
            ("Overview", "caselens_page1_overview.png"),
            ("Funnel", "caselens_page2_funnel.png"),
            ("Duration", "caselens_page3_bottlenecks.png"),
            ("Linkage", "caselens_page4_linkage.png"),
            ("Before", "caselens_page5_before_after.png"),
            ("Survivor Privacy", "caselens_page6_privacy.png"),
            ("Statutory", "caselens_page7_statutory.png")
        ]
        for query, fname in pages:
            asyncio.run(capture_page_screenshot(fname, query))

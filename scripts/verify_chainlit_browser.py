"""Exercise the local Chainlit UI with Playwright and save a screenshot and result."""

import argparse
import json
from pathlib import Path


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expect-offline", action="store_true", help="Verify missing-endpoint handling; does not validate model predictions.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "data/session4"
    output.mkdir(parents=True, exist_ok=True)
    sentence = "A siege will crack the strong defense."
    expected = "The translation model is offline." if args.expect_offline else "Crack the strong defense, a siege will."
    with sync_playwright() as playwright:
        options = {"headless": True}
        if Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe").is_file():
            options["channel"] = "msedge"
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Send an English sentence, and I’ll rewrite it in Yoda’s style.", exact=False).wait_for(timeout=30000)
        editor = page.locator("textarea, [contenteditable='true']").first
        editor.fill(sentence)
        editor.press("Enter")
        page.get_by_text(expected, exact=False).last.wait_for(timeout=180000)
        screenshot = output / ("chainlit_offline.png" if args.expect_offline else "chainlit_chat.png")
        page.screenshot(path=str(screenshot), full_page=True)
        result = {"mode": "offline-error-flow" if args.expect_offline else "live-prediction",
                  "url": args.url, "sentence": sentence, "expected_response_fragment": expected,
                  "screenshot": str(screenshot.relative_to(root)), "verified": True}
        (output / "browser_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        browser.close()


if __name__ == "__main__":
    main()

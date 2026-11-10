"""Exercise the local Chainlit UI with Playwright and save a screenshot and result."""

import argparse
import json
from pathlib import Path


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--task", choices=["restaurant", "yoda"], default="restaurant")
    parser.add_argument("--expected-response", help="Expected live response; restaurant mode otherwise reads the saved API check.")
    parser.add_argument("--expect-offline", action="store_true", help="Verify missing-endpoint handling; does not validate model predictions.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / ("data/restaurant" if args.task == "restaurant" else "data/session4")
    output.mkdir(parents=True, exist_ok=True)
    sentence = json.loads((root / "examples/restaurants.json").read_text(encoding="utf-8"))[0]["attributes"] if args.task == "restaurant" else "A siege will crack the strong defense."
    if args.expect_offline:
        expected = "The model is offline."
    elif args.expected_response:
        expected = args.expected_response
    elif args.task == "restaurant":
        response = json.loads((output / "online_prediction.json").read_text(encoding="utf-8"))
        expected = response.get("response") or response["predictions"][0]["generated_text"]
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("Run test_online_endpoint.py with the default restaurant fixture first.")
    else:
        expected = "Crack the strong defense, a siege will."
    with sync_playwright() as playwright:
        options = {"headless": True}
        if Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe").is_file():
            options["channel"] = "msedge"
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.url, wait_until="networkidle")
        welcome = "Send restaurant attributes" if args.task == "restaurant" else "Send an English sentence"
        page.get_by_text(welcome, exact=False).wait_for(timeout=30000)
        editor = page.locator("textarea, [contenteditable='true']").first
        editor.fill(sentence)
        editor.press("Enter")
        page.get_by_text(expected, exact=False).last.wait_for(timeout=180000)
        screenshot = output / ("chainlit_offline.png" if args.expect_offline else "chainlit_chat.png")
        page.screenshot(path=str(screenshot), full_page=True)
        result = {"mode": "offline-error-flow" if args.expect_offline else "live-prediction",
                  "task": args.task,
                  "url": args.url, "sentence": sentence, "expected_response_fragment": expected,
                  "screenshot": str(screenshot.relative_to(root)), "verified": True}
        (output / "browser_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        browser.close()


if __name__ == "__main__":
    main()

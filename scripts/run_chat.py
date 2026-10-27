"""Start the local Chainlit app with a predictable working directory and CLI environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Chainlit interprets DEBUG as a boolean. The surrounding IDE may set it to 'release'.
    environment = dict(os.environ)
    environment["DEBUG"] = "false"
    command = [sys.executable, "-m", "chainlit", "run", "src/agentic_workflows/app/main.py",
               "--headless", "--host", "127.0.0.1", "--port", str(args.port)]
    try:
        raise SystemExit(subprocess.call(command, cwd=root, env=environment))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()

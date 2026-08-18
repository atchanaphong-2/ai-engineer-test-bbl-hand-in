"""Single entry point: builds AppContainer and serves API + frontend together."""

import logging
import sys

import uvicorn

from agentic_rag.backend.main import create_app
from agentic_rag.bootstrap import AppContainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    try:
        container = AppContainer()
        app = create_app(container)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    uvicorn.run(app, host=container.settings.host, port=container.settings.port)


if __name__ == "__main__":
    main()

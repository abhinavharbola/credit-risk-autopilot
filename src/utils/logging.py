import os
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

_configured = False


def configure_logging(service_name: str = "credit-risk-governance") -> None:
    global _configured
    if _configured:
        return

    try:
        import logfire

        token = os.environ.get("LOGFIRE_TOKEN") or None
        logfire.configure(
            token=token,
            service_name=service_name,
            send_to_logfire=bool(token),
        )
        _configured = True
    except Exception:
        return


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    if not _configured:
        yield
        return

    import logfire

    with logfire.span(name, **attributes):
        yield

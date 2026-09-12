"""Logfire setup for tracing pipeline runs and serving requests.

configure_logging() is idempotent: only the first call in a process
actually configures the global Logfire instance, so every entrypoint
(scripts/advance_clock.py, scripts/run_demo_loop.py, src/serving/app.py)
can call it unconditionally on startup without worrying about double
configuration.

LOGFIRE_TOKEN is optional, and so is the `logfire` package itself, along
with the package simply working at all: configure_logging() catches any
failure during import or configure, not just a missing package, and
no-ops. Tracing is a nice-to-have, not something that should block
starting the serving app or the demo loop.
"""

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
        # Any failure here (missing package, or a package that's installed
        # but broken on this platform - logfire 2.6.1 calls
        # os.register_at_fork() at import time, which doesn't exist on
        # Windows) should not be able to take down serving startup or the
        # demo loop over an optional tracing dependency. Caught broadly on
        # purpose: this is third-party init code we don't control, not
        # something we can narrow to one expected exception type.
        return


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Thin wrapper so callers don't need to import logfire directly or care
    whether configure_logging() has already run. Falls back to a plain
    no-op context manager if logging hasn't been configured yet (e.g.
    running tests offline, where logfire isn't installed at all).
    """
    if not _configured:
        yield
        return

    import logfire

    with logfire.span(name, **attributes):
        yield

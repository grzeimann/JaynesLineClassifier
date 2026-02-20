from __future__ import annotations

from datetime import datetime


def timestamp() -> str:
    """Return local time as 'YYYY-MM-DD HH:MM:SS.mmm' (milliseconds, no timezone).

    We avoid the ISO 8601 'T' separator and timezone offset to keep logs compact
    while remaining sortable and now include millisecond precision.
    """
    try:
        # Use microseconds from strftime and trim to milliseconds
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        return ts[:-3]  # strip to milliseconds
    except Exception:
        # Fallbacks if strftime or locale settings fail
        try:
            # Python 3.6+: isoformat with timespec may not be available everywhere
            try:
                return datetime.now().isoformat(timespec="milliseconds")
            except TypeError:
                # Older Pythons: full isoformat (may include microseconds)
                return datetime.now().isoformat()
        except Exception:
            return "unknown-time"


def log(message: str) -> None:
    """Print a message prefixed with a timestamp.

    Minimal helper to add a time element to existing print-style logs without
    introducing a full logging framework.
    """
    try:
        ts = timestamp()
        print(f"[{ts}] {message}")
    except Exception:
        # Be robust if stdout or encoding is unavailable
        try:
            print(message)
        except Exception:
            pass

"""
api_retry.py — Retry and Exponential Backoff Utility for API Calls (v2.0)
=========================================================================
Provides robust retry handling with intelligent 429 RateLimit backoff.
"""

import time
import re
import functools
from typing import Callable, Any, Type


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> Any:
    """
    Executes func(*args, **kwargs) with exponential backoff retries.
    If a 429 RateLimitError is detected with a recommended retryDelay, it waits for that delay.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exception = exc
            if attempt == max_retries:
                print(f"  ⚠ [api_retry] All {max_retries} attempts failed: {exc}")
                raise exc

            # Check for 429 RateLimit retryDelay suggestion in error message
            err_msg = str(exc)
            retry_match = re.search(r"retry(?:ing)? in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE) or \
                          re.search(r"retryDelay':\s*'(\d+)s'", err_msg, re.IGNORECASE)

            if retry_match:
                suggested_wait = float(retry_match.group(1)) + 1.0
                delay = max(delay, min(suggested_wait, 30.0))  # cap max wait at 30s

            if on_retry:
                on_retry(exc, attempt, delay)
            else:
                print(f"  ⚠ [api_retry] Attempt {attempt}/{max_retries} failed ({exc}). Waiting {delay:.1f}s before retry…")

            time.sleep(delay)
            delay *= backoff_factor

    if last_exception:
        raise last_exception

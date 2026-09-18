# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Rate Limiter Utility

Provides consistent rate limiting across all pipeline components
to avoid AWS Bedrock throttling exceptions.
"""

import time
import random
import threading
from typing import Optional


class RateLimiter:
    """Simple rate limiter with configurable delays (thread-safe)."""

    def __init__(self, base_delay: float = 0.5, jitter: bool = True):
        """
        Initialize rate limiter.

        Args:
            base_delay: Base delay in seconds between calls
            jitter: Whether to add random jitter to avoid synchronized requests
        """
        self.base_delay = base_delay
        self.jitter = jitter
        self.last_call_time = 0
        self._lock = threading.Lock()

    def wait(self, custom_delay: Optional[float] = None):
        """
        Wait for the appropriate amount of time before next API call.

        Args:
            custom_delay: Override the base delay for this call
        """
        delay = custom_delay if custom_delay is not None else self.base_delay

        if self.jitter:
            # Add 0-20% jitter to avoid synchronized requests
            jitter_amount = delay * 0.2 * random.random()
            delay += jitter_amount

        # Serialize across threads so concurrent workers are actually spaced by `delay`
        # (prevents throttle bursts under parallel evaluation).
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < delay:
                sleep_time = delay - time_since_last_call
                time.sleep(sleep_time)

            self.last_call_time = time.time()

    def wait_between_items(self, current_index: int, total_items: int, custom_delay: Optional[float] = None):
        """
        Wait between processing items, but not after the last item.

        Args:
            current_index: Current item index (0-based)
            total_items: Total number of items
            custom_delay: Override the base delay for this call
        """
        if current_index < total_items - 1:  # Don't wait after the last item
            self.wait(custom_delay)


# Global rate limiter instance
_global_rate_limiter = RateLimiter()


def wait_for_rate_limit(delay: Optional[float] = None):
    """
    Convenience function to wait for rate limiting.

    Args:
        delay: Custom delay in seconds (uses default if None)
    """
    _global_rate_limiter.wait(delay)


def wait_between_items(current_index: int, total_items: int, delay: Optional[float] = None):
    """
    Convenience function to wait between processing items.

    Args:
        current_index: Current item index (0-based)
        total_items: Total number of items
        delay: Custom delay in seconds (uses default if None)
    """
    _global_rate_limiter.wait_between_items(current_index, total_items, delay)


def configure_rate_limiter(base_delay: float = 0.5, jitter: bool = True):
    """
    Configure the global rate limiter.

    Args:
        base_delay: Base delay in seconds between calls
        jitter: Whether to add random jitter
    """
    global _global_rate_limiter
    _global_rate_limiter = RateLimiter(base_delay, jitter)

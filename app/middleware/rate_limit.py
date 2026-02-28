"""
Rate limiting configuration using slowapi.

Limits are configurable via ``RATE_LIMIT_CHAT`` and ``RATE_LIMIT_GENERAL``
environment variables (e.g. ``10/minute``, ``100/hour``).
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# The limiter singleton — imported and used by route modules.
limiter = Limiter(key_func=get_remote_address)

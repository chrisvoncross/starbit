"""One sealed hour in hand, and the batches made from its candles.

The stream reads. The feed batches. Nothing here scales or normalizes.
"""

from .feed.cursor import Feed
from .logger import Log
from .read.stream import Stream

__all__ = ["Feed", "Log", "Stream"]

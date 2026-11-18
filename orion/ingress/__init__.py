"""One sealed hour in hand, and the batches made from its candles.

The stream reads. The feed batches. Nothing here scales or normalizes.
"""

from .feed.cursor import Feed
from .logger import Log
from .read.deals import DealStream, open_deals
from .read.stream import Stream

__all__ = ["DealStream", "Feed", "Log", "Stream", "open_deals"]

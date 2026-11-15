"""A high-low stretch is counted without the indicator. A gap breaks it."""

from orion.models.nova.ride import Ride
from orion.models.nova.spec import CANDLE_MS, HIGH, LOW, NONE


def test_a_flip_stretch_counts_without_matching_the_indicator():
    ride = Ride()
    mark = ride.mark()
    stamp = 0
    sides = (HIGH, LOW, HIGH, LOW)
    for side in sides:
        ride.see(130, stamp, side, NONE, 1.2, 1.0, 0.1)
        stamp += CANDLE_MS
    report = ride.batch(mark)
    assert ride.max_ride == 4
    assert ride.max_hit == 0
    assert report["off"] == 1.0
    assert report["ride"] == 3 / 4
    assert report["off_ride"] == 3 / 4


def test_a_repeat_a_narrow_candle_and_a_gap_do_not_extend():
    ride = Ride()
    stamp = 0
    ride.see(130, stamp, HIGH, HIGH, 1.2, 1.0, 0.1)
    stamp += CANDLE_MS
    ride.see(130, stamp, HIGH, LOW, 1.2, 1.0, 0.1)
    stamp += CANDLE_MS
    ride.see(130, stamp, LOW, LOW, 1.05, 1.0, 0.1)
    stamp += 2 * CANDLE_MS
    ride.see(130, stamp, HIGH, HIGH, 1.2, 1.0, 0.1)
    assert ride.max_ride == 1
    assert ride.ride_steps == 0
    assert ride.max_hit == 1
    assert ride.correct == 3

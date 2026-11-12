"""Hit counts. No network."""

from __future__ import annotations

import numpy as np

from orion.models.nova.log import Tally, event_fields


def test_marks_are_counted_apart_from_the_other_rows():
    tally = Tally()
    tally.add(
        np.array([0, 1, 2]),
        np.array([0, 0, 2]),
        np.array([True, True, False]),
        np.array([[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6], [9.0, 9.0, 9.0, 9.0]]),
    )
    fields = tally.fields()
    assert fields["class_hit"] == 2 / 3
    assert fields["mark_hit"] == 0.5
    assert fields["call_share"] == 1 / 3
    assert np.isclose(fields["sketch_high"], 0.3)
    assert np.isclose(fields["mark_sketch_high"], 0.4)
    assert np.isclose(fields["mark_sketch_low"], 0.5)
    assert event_fields(fields)["tally_marks"] == 2
    assert "rows" not in event_fields(fields)
    copy = Tally()
    copy.restore(tally.export())
    tally.add(
        np.array([1]),
        np.array([1]),
        np.array([True]),
        np.array([[0.0, 0.0, 0.0, 0.0]]),
    )
    assert copy.fields()["marks"] == 2
    assert tally.fields()["marks"] == 3

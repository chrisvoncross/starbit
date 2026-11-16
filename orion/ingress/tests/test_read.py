"""Decode, listing, and the 204 to 216 expansion. No network."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

from orion.ingress.read.decode import DecodeError, ints, numbers
from orion.ingress.read.frame import HourError, expand_features, hour_frames, read_hour
from orion.ingress.read.lake import LakeError, enumerate_partitions, list_sealed_hours, open_lance
from orion.ingress.read.lanes import FEATURE_DIM, GROUP_SLICES
from orion.ingress.tests.tables import BASE, STEP, flat, raises, table

_ROOT = Path(__file__).resolve().parents[1]


def test_feature_width_other_than_204_or_216_raises():
    bad = table([flat(2, 0), flat(2, STEP)], width=8)
    exc = raises(HourError, lambda: hour_frames(bad))
    assert "8" in str(exc)
    assert raises(HourError, lambda: expand_features(np.zeros((2, 100)), np.zeros(2, dtype=np.int64), {}))


def test_a_204_vector_grows_to_216_and_a_216_vector_passes_through():
    void = 2**32 - 1
    rows = [flat(2, 0), flat(2, STEP, BASE + 100)]
    extra = {
        "bar_kind": pa.array([0, 3], pa.int8()),
        "trade_count": pa.array([4, 0], pa.int64()),
        "aux_seen": pa.array([0b01000010, 0], pa.uint8()),
        "funding_staleness_ms": pa.array([10, void], pa.uint32()),
        "ticker_24h_change_fp": pa.array([100_000_000, 100_000_000], pa.int64()),
        "vwap_fp": pa.array([110_000_000, 110_000_000], pa.int64()),
    }
    # close_fp is BASE (1.0) on the first row and the price column is also the close.
    built, trades = hour_frames(table(rows, extra=extra))
    features = built[0].features
    assert features.shape == (2, FEATURE_DIM)
    since = GROUP_SLICES["awareness"][0]
    assert np.isnan(features[0, since])
    assert features[1, since] == np.log1p(STEP / 1000.0)
    assert features[0, since + 1] == 0.0
    assert features[1, since + 1] == 1.0
    projection = GROUP_SLICES["projection"][0]
    assert features[0, projection] == 10.0
    assert np.isnan(features[1, projection])
    assert features[0, projection + 4] == 1.0  # depth bit 1
    assert features[0, projection + 5] == 1.0  # hold bit 6
    assert features[0, projection + 6] == 1.0  # ticker / 1e8
    assert features[0, projection + 7] == BASE / 1e8  # volume, not scaled again
    assert features[0, projection + 8] == 4.0  # trade_count kept
    assert features[1, projection + 8] == 0.0
    assert trades[2] == 0  # the boundary bar is not a real trade

    wide = np.zeros((2, FEATURE_DIM), dtype=np.float64)
    wide[:, since + 1] = 0.25
    kept = expand_features(wide, np.asarray([0, STEP]), {"bar_kind": np.asarray([3.0, 3.0])})
    assert kept.shape == (2, FEATURE_DIM)
    assert np.all(kept[:, since + 1] == 0.25)


def test_integer_prices_are_divided_and_floats_stay_floats():
    rows = [flat(2, 0, 250_000_000)]
    built, _trades = hour_frames(table(rows))
    assert built[0].close[0] == 2.5
    floated = table(rows)
    floated = floated.set_column(
        floated.schema.get_field_index("close_fp"),
        "close_fp",
        pa.array([1.25], pa.float64()),
    )
    built, _trades = hour_frames(floated)
    assert built[0].close[0] == 1.25


def test_dictionary_and_null_keys_raise_and_value_dictionaries_decode():
    bad_key = table([flat(2, 0)])
    encoded = bad_key.column("symbol").dictionary_encode()
    bad_key = bad_key.set_column(bad_key.schema.get_field_index("symbol"), "symbol", encoded)
    exc = raises(HourError, lambda: hour_frames(bad_key))
    assert "dictionary" in str(exc).lower()
    assert raises(DecodeError, lambda: ints(encoded, "symbol"))

    null_key = table([flat(2, 0), flat(2, STEP)])
    null_key = null_key.set_column(
        null_key.schema.get_field_index("open_time_ms"),
        "open_time_ms",
        pa.array([0, None], pa.int64()),
    )
    exc = raises(HourError, lambda: hour_frames(null_key))
    assert "null" in str(exc).lower()

    text = table([flat(2, 0)])
    text = text.set_column(text.schema.get_field_index("symbol"), "symbol", pa.array(["btc"]))
    exc = raises(HourError, lambda: hour_frames(text))
    assert "text" in str(exc).lower()

    encoded_price = pa.array([100_000_000], pa.int64()).dictionary_encode()
    assert numbers(encoded_price, "close_fp", fixed=True)[0] == 1.0
    assert numbers(pa.array([5, None], pa.int64()), "trade_count", fixed=False).tolist()[0] == 5.0
    assert np.isnan(numbers(pa.array([5, None], pa.int64()), "trade_count", fixed=False)[1])
    flags = numbers(pa.array([True, False, None]), "flag", fixed=False)
    assert flags[0] == 1.0 and flags[1] == 0.0 and np.isnan(flags[2])
    exc = raises(DecodeError, lambda: numbers(pa.array(["x"]), "volume_fp", fixed=True))
    assert "volume_fp" in str(exc)


def test_duplicate_timestamp_keeps_the_earlier_row_and_a_gap_stays():
    earlier = BASE
    later = BASE + 50_000
    rows = [
        (2, 0, earlier, earlier, earlier),
        (2, 0, later, later, later),
        flat(2, STEP),
        flat(2, 3 * STEP),
    ]
    built, _trades = hour_frames(table(rows))
    assert built[0].open_time_ms.tolist() == [0, STEP, 3 * STEP]
    assert built[0].close[0] == earlier / 1e8


def test_read_hour_asks_only_for_the_columns_it_uses():
    stored = table([flat(2, 0)], extra={"bar_kind": pa.array([0], pa.int8()), "ignored": pa.array([1], pa.int64())})

    class Dataset:
        schema = stored.schema

        def __init__(self):
            self.calls = []

        def to_table(self, columns, filter=None):
            self.calls.append((list(columns), filter))
            return stored.select(columns)

    dataset = Dataset()
    found = read_hour("mem://hour", opener=lambda _source: dataset)
    assert len(dataset.calls) == 1
    assert "ignored" not in dataset.calls[0][0]
    assert "bar_kind" in dataset.calls[0][0]
    assert "feature_vec" in dataset.calls[0][0]
    assert found.num_rows == 1
    bare = type("Bare", (), {"schema": pa.schema([("symbol", pa.int64())]), "to_table": lambda *a, **k: (_ for _ in ()).throw(AssertionError("read"))})()
    raises(HourError, lambda: read_hour("mem://hour", opener=lambda _source: bare))


def test_read_hour_drops_a_leaked_symbol_before_feature_vec():
    from orion.ingress.read import frame as frame_mod

    stored = table([flat(2, 0), flat(9, 0)])

    class Dataset:
        schema = stored.schema

        def to_table(self, columns, filter=None):
            assert filter == "symbol IN (2)"
            return stored.select(columns)

    found = read_hour("mem://hour", opener=lambda _source: Dataset(), symbols=(2,))
    assert found.column("symbol").to_pylist() == [2]
    lengths = []
    real = frame_mod.decode_feature_vec

    def wrapped(column):
        lengths.append(len(column))
        return real(column)

    frame_mod.decode_feature_vec = wrapped
    try:
        built, _trades = hour_frames(found)
    finally:
        frame_mod.decode_feature_vec = real
    assert lengths == [1]
    assert [series.symbol for series in built] == [2]


def test_listing_reads_names_and_open_takes_credentials_per_call(tmp_path: Path):
    keep = tmp_path / "p=2026-10-06T09"
    keep.mkdir()
    (keep / "dataset.lance").write_bytes(b"")
    (tmp_path / "p=2026-10-06T08").mkdir()
    (tmp_path / "notes").mkdir()
    found = list_sealed_hours(tmp_path)
    assert [part.hour for part in found] == ["2026-10-06T09"]
    assert found[0].row_count == 0
    assert found[0].uri == str((keep / "dataset.lance").resolve())
    target = tmp_path / "hour"
    target.write_text("nope", encoding="utf-8")
    raises(LakeError, lambda: list_sealed_hours(target))

    class Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return self

        def paginate(self, *, Bucket, Prefix, Delimiter):
            assert Bucket == "bucket" and Prefix == "root/" and Delimiter == "/"
            return [
                {
                    "CommonPrefixes": [
                        {"Prefix": "root/p=2026-10-06T10/"},
                        {"Prefix": "root/p=2026-10-06T09/"},
                        {"Prefix": "root/not-an-hour/"},
                    ]
                }
            ]

    listed = enumerate_partitions(bucket="bucket", prefix="root", client=Client())
    assert [part.hour for part in listed] == ["2026-10-06T09", "2026-10-06T10"]
    assert listed[0].uri == "s3://bucket/root/p=2026-10-06T09/dataset.lance"

    seen = []

    def opener(source):
        seen.append(source)
        return "dataset"

    assert open_lance("s3://bucket/p=2026-10-06T09/dataset.lance", opener=opener) == "dataset"
    assert seen == ["s3://bucket/p=2026-10-06T09/dataset.lance"]
    raises(FileNotFoundError, lambda: open_lance(tmp_path / "missing.lance"))

    import lance

    captured = []
    original = lance.dataset

    def fake(uri, storage_options=None):
        captured.append((uri, dict(storage_options or {})))
        raise FileNotFoundError(uri)

    keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_DEFAULT_REGION")
    saved = {key: os.environ.get(key) for key in keys}
    lance.dataset = fake
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = "abc"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret-a"
        os.environ["AWS_ENDPOINT_URL"] = "https://example.test"
        os.environ["AWS_DEFAULT_REGION"] = "auto"
        raises(FileNotFoundError, lambda: open_lance("s3://bucket/one"))
        os.environ["AWS_ACCESS_KEY_ID"] = "def"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret-b"
        raises(FileNotFoundError, lambda: open_lance("s3://bucket/two"))
    finally:
        lance.dataset = original
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert captured[0][1]["AWS_ACCESS_KEY_ID"] == "abc"
    assert captured[1][1]["AWS_ACCESS_KEY_ID"] == "def"
    assert captured[0][1]["AWS_SECRET_ACCESS_KEY"] == "secret-a"
    assert captured[1][1]["AWS_SECRET_ACCESS_KEY"] == "secret-b"


def test_the_package_does_not_import_the_other_trees():
    banned_prefix = ("altes", "stardust", "horizon", "torch", "boto3", "dali", "mosaic")
    for path in _ROOT.rglob("*.py"):
        if "tests" in path.parts or path.name == "README.md":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                head = name.split(".")[0]
                assert head not in banned_prefix, f"{path.name} imports {name}"
                assert not name.startswith("orion.data"), f"{path.name} imports {name}"
                assert not (name == "orion" or (name.startswith("orion.") and not name.startswith("orion.ingress")))
            if "read" in path.parts and isinstance(node, ast.ImportFrom):
                assert "feed" not in (node.module or "").split(".")

"""Forms, the tick, and the three answers. The manifest picks the form."""

from __future__ import annotations

from dataclasses import dataclass

LANES = 216
PRICES = 4
CANDLE_MS = 15_000
HOUR_MS = 3_600_000
HOUR_CANDLES = 240
TAIL = 20
TICK = 0.1
NONE = 0
LOW = 1
HIGH = 2
NEXT_BINS = 128
EXPAND = 2
MAMBA_SSM = "2.3.2.post1"
SAMPLE_EVERY = 64


@dataclass(frozen=True)
class Form:
    """One width, one depth. Head count is the official Mamba-3 count."""

    name: str
    width: int
    layers: int
    heads: int
    headdim: int
    d_state: int

    def __post_init__(self) -> None:
        if self.d_state % 2 != 0:
            raise ValueError("d_state must be even so the angle can rotate pairs")
        if self.layers < 1 or self.width < 1 or self.headdim < 1:
            raise ValueError("form sizes must be positive")
        if self.headdim & (self.headdim - 1) or self.d_state & (self.d_state - 1):
            raise ValueError("the SISO kernel needs headdim and d_state as powers of two")
        official = (EXPAND * self.width) // self.headdim
        if (EXPAND * self.width) % self.headdim != 0 or self.heads != official:
            raise ValueError("heads must match Mamba3: expand * width / headdim")


SMALL = Form("small", 64, 4, 4, 32, 64)
STUDY = Form("study", 256, 13, 8, 64, 128)
FORMS = {"small": SMALL, "study": STUDY}


def by_name(name: str) -> Form:
    try:
        return FORMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown form {name!r}") from exc


@dataclass(frozen=True)
class Manifest:
    """Written before any hour opens. One loss per run."""

    form: str
    loss: str
    lookback: int
    batch: int
    seed: int
    steps: int
    ckpt_every: int
    lr: float
    train_days: tuple[str, ...] = ()
    test_days: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.loss not in ("side", "next"):
            raise ValueError("a run has one loss: side or next")
        if self.lookback < 1 or self.batch < 1 or self.steps < 1 or self.ckpt_every < 1:
            raise ValueError("manifest sizes must be positive")
        by_name(self.form)

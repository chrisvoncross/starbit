"""The first run named in docs/run.md. One day, one symbol, the small form."""

from orion.models.door.hours import dotenv, hours_of, open_hour
from orion.models.nova.run import run
from orion.models.nova.spec import Manifest

TRAIN = ("2026-10-07",)
LOG = "orion/models/nova/docs/runs/first.jsonl"
CKPT = "orion/models/nova/docs/runs/first"


def read(token):
    return open_hour(token, symbols=(130,))


def main() -> None:
    dotenv()
    manifest = Manifest(
        form="small",
        loss="side",
        lookback=240,
        batch=8,
        seed=0,
        steps=4000,
        ckpt_every=50,
        lr=1e-4,
        train_days=TRAIN,
        test_days=(),
    )
    run(manifest, hours_of(TRAIN), read=read, log_path=LOG, ckpt_dir=CKPT)


if __name__ == "__main__":
    main()

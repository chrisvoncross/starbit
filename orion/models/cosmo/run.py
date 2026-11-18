"""Write, test, derive. Learn hours and later hours come from the start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion.models.cosmo.derive.next import instruction
from orion.models.cosmo.read.load import samples
from orion.models.cosmo.record.log import Record
from orion.models.cosmo.test.check import compile_predict, keep
from orion.models.cosmo.write.ask import ask, post


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmo: write a function, test it on later hours.")
    parser.add_argument("--learn", required=True, help="Hour tokens, comma separated.")
    parser.add_argument("--later", required=True, help="Later hour tokens, comma separated.")
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--symbols", default="", help="Symbol ids, comma separated. Empty means every symbol in the hours.")
    parser.add_argument("--deals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, poster=None, read=None, deal_read=None) -> int:
    args = parse(argv)
    if args.rounds < 1:
        raise SystemExit("rounds must be at least one")
    learn_hours = _hours(args.learn)
    later_hours = _hours(args.later)
    symbols = _symbols(args.symbols)
    root = Path(args.run_dir)
    record = Record(root / "record.jsonl")
    kept_path = root / "kept.jsonl"
    rejected_path = root / "rejected.jsonl"
    model_call = poster or _remote(args.model_url, args.model)
    kept = _load(kept_path)
    rejected = _load(rejected_path)
    sequence = [_ready(item) for item in kept]
    lookback = max((item["candles"] for item in sequence), default=1)
    record.write("start", {"learn": learn_hours, "later": later_hours, "deals": args.deals, "symbols": sorted(symbols) if symbols is not None else None}, {"rounds": args.rounds})
    for _ in range(args.rounds):
        note = instruction(kept, rejected)
        record.write("derive", {"kept": len(kept), "rejected": len(rejected)}, note)
        item, failure = _written(model_call, note, record)
        if item is None:
            record.write("test", {"source": failure}, {"executed": False})
            continue
        lookback = max(lookback, item["candles"])
        learn = samples(learn_hours, lookback=lookback, deals=args.deals, symbols=symbols, read=read, deal_read=deal_read)
        later = samples(later_hours, lookback=lookback, deals=args.deals, symbols=symbols, read=read, deal_read=deal_read)
        record.write("read", {"learn_hours": learn_hours, "later_hours": later_hours, "candles": item["candles"]}, {"learn_rows": len(learn), "later_rows": len(later)})
        try:
            ready = _ready(item)
        except (SyntaxError, ValueError) as exc:
            record.write("test", {"name": item["name"]}, {"executed": False, "error": str(exc)})
            continue
        accepted = keep(sequence, ready, learn, later)
        store = kept_path if accepted else rejected_path
        _append(store, item)
        record.write("test", {"name": item["name"], "candles": item["candles"]}, {"kept": accepted, "later_rows": len(later)})
        if accepted:
            kept.append(item)
            sequence.append(ready)
        else:
            rejected.append(item)
    return 0


def _written(model_call, note: str, record: Record):
    try:
        item = ask(model_call, note)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return None, str(exc)
    record.write("write", {"instruction": note}, {"name": item["name"], "candles": item["candles"]})
    return item, None


def _ready(item: dict) -> dict:
    return {"name": item["name"], "candles": item["candles"], "predict": compile_predict(item["source"])}


def _remote(url: str, model: str):
    name = model or _first_model(url)

    def poster(text: str) -> str:
        return post(url, name, text)

    return poster


def _first_model(url: str) -> str:
    with urllib_request(url) as response:
        body = json.load(response)
    return body["data"][0]["id"]


def urllib_request(url: str):
    import urllib.request

    return urllib.request.urlopen(url.rstrip("/") + "/models")


def _hours(text: str) -> tuple[str, ...]:
    hours = tuple(part.strip() for part in text.split(",") if part.strip())
    if not hours:
        raise SystemExit("hour list is empty")
    return hours


def _symbols(text: str) -> set[int] | None:
    if not text.strip():
        return None
    return {int(part) for part in text.split(",") if part.strip()}


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"name": item["name"], "candles": item["candles"], "source": item["source"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

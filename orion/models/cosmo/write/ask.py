"""One model call. The reply is a function and the candles it reads."""

from __future__ import annotations

import json
import urllib.request

_ASK = """Gib eine Python-Funktion und die Zahl der Kerzen, die sie bis zum Schluss liest.
Die Funktion heißt predict. Sie bekommt candles und deals, die schon geladen sind.
candles hat time, open, high, low, close, volume, features. Die letzte Zeile ist dieser Schluss.
deals hat time, price, qty, side. Deals der nächsten Kerze sind nicht dabei.
predict gibt (hochabstand, tiefabstand) zurück, oder None wenn diese Bedingung die Zeile nicht trifft.
Sie öffnet keine Datei, kein Netz und keine weitere Stunde. Nur diese geladenen Reihen.
Antworte nur mit JSON: {"name": "...", "candles": 1, "source": "def predict(candles, deals):\\n    return None\\n"}
"""


def ask(poster, instruction: str) -> dict:
    """``poster`` is the model call. Tests pass a function. A run passes ``post``."""

    text = poster(_ASK + "\n" + instruction)
    body = _json(text)
    source = str(body["source"])
    if "def predict" not in source:
        raise ValueError("the reply has no predict function")
    candles = body.get("candles", 1)
    if isinstance(candles, bool) or not isinstance(candles, int) or candles < 1:
        raise ValueError("candles must count the close itself")
    return {"name": str(body.get("name", "predict")), "candles": candles, "source": source}


def post(url: str, model: str, text: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        reply = json.load(response)
    return reply["choices"][0]["message"]["content"]


def _json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the reply is not JSON")
    return json.loads(raw[start : end + 1])

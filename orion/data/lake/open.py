"""Open one Lance dataset.

Credentials come from the environment for this call and are not stored.
Tests pass ``opener`` and do not touch the network. An absent dataset is
``FileNotFoundError``. Every other failure propagates unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REMOTE_SCHEMES = ("s3://", "r2://", "gs://", "az://", "http://", "https://")
_REMOTE_ABSENT = ("DatasetNotFound", "NoSuchKey", "was not found", "Not found:")


def is_absent(exc: BaseException) -> bool:
    """True only for a missing dataset. Auth and other failures stay failures."""

    if isinstance(exc, FileNotFoundError):
        return True
    message = str(exc)
    return any(mark in message for mark in _REMOTE_ABSENT)


def is_remote(uri: str) -> bool:
    return any(uri.startswith(scheme) for scheme in _REMOTE_SCHEMES)


def _storage_options() -> dict[str, str]:
    opts: dict[str, str] = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    if endpoint:
        opts["AWS_ENDPOINT_URL"] = endpoint
    access = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID")
    if access:
        opts["AWS_ACCESS_KEY_ID"] = access
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY")
    if secret:
        opts["AWS_SECRET_ACCESS_KEY"] = secret
    region = os.environ.get("AWS_DEFAULT_REGION") or ("auto" if endpoint else None)
    if region:
        opts["AWS_DEFAULT_REGION"] = region
    return opts


def open_lance(uri: str, *, opener: Any = None):
    """Open ``uri``. ``FileNotFoundError`` means the dataset is absent.

    ``opener(uri)`` replaces the Lance call. A remote absence is recognised
    only by Lance's not-found messages; auth and network errors raise as
    they are.
    """
    text = os.fspath(uri)
    if opener is not None:
        return opener(text)
    import lance

    if not is_remote(text):
        if not Path(text).exists():
            raise FileNotFoundError(text)
        return lance.dataset(text)
    try:
        return lance.dataset(text, storage_options=_storage_options() or None)
    except Exception as exc:
        if is_absent(exc):
            raise FileNotFoundError(text) from exc
        raise

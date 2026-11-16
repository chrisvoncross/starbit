"""Open one Lance dataset and list sealed-hour names.

Credentials are read from the environment for this call and are not stored.
Listing reads directory or prefix names only. It does not open a dataset
and it does not read candles. Tests pass ``opener`` or ``client`` and do
not touch the network.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import Any

_REMOTE_SCHEMES = ("s3://", "r2://", "gs://", "az://", "http://", "https://")
_REMOTE_ABSENT = ("DatasetNotFound", "NoSuchKey", "was not found", "Not found:")
_AUTH_MARKS = (
    "AccessDenied",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "ExpiredToken",
    "NotAuthorized",
    "Unauthorized",
    "Forbidden",
    "InvalidToken",
    "NoCredentials",
    "AuthFailure",
    "AccessDeniedException",
)
_PARTITION_NAME = re.compile(r"^p=(.+)$")
_PARTITION_DIR = re.compile(r"p=([^/]+)/?$")
_DATASET_LEAF = "dataset.lance"


class LakeError(ValueError):
    """A sealed-hour path could not be listed."""


@dataclasses.dataclass(frozen=True)
class PartitionInfo:
    """One sealed hour. ``hour`` is the token inside ``p=<hour>``.

    ``row_count`` stays 0. Listing does not read the file.
    """

    hour: str
    uri: str
    row_count: int = 0


def is_remote(uri: str) -> bool:
    return any(uri.startswith(scheme) for scheme in _REMOTE_SCHEMES)


def is_absent(exc: BaseException) -> bool:
    """True only for a missing dataset. Auth and other failures stay failures."""
    if isinstance(exc, FileNotFoundError):
        return True
    message = str(exc)
    return any(mark in message for mark in _REMOTE_ABSENT)


def is_auth(exc: BaseException) -> bool:
    """An access failure. It is not a missing file and it is not retried."""
    if isinstance(exc, FileNotFoundError):
        return False
    if isinstance(exc, PermissionError):
        return True
    text = f"{type(exc).__name__} {exc}"
    return any(mark in text for mark in _AUTH_MARKS)


def _storage_options() -> dict[str, str]:
    """Credentials for this call only. The dict is not kept."""
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


def open_lance(uri: str | os.PathLike[str], *, opener: Any = None):
    """Open ``uri``. ``FileNotFoundError`` means the dataset is absent.

    ``opener(uri)`` replaces the Lance call. A remote absence is recognised
    only by Lance's not-found messages. Auth and other errors propagate.
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


def enumerate_partitions(
    *,
    bucket: str,
    prefix: str,
    client: Any,
) -> list[PartitionInfo]:
    """Hour tokens under an S3 or R2 prefix, sorted by token.

    The layout is ``<prefix>p=<hour>/dataset.lance``. ``client`` is an S3
    client. This package does not build one. Tests pass a fake. The call
    lists names. It does not open a dataset.
    """
    if client is None:
        raise LakeError("enumerate_partitions needs a client")
    clean = prefix.rstrip("/") + "/"
    partitions: list[PartitionInfo] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=clean, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            dir_key = entry["Prefix"]
            match = _PARTITION_DIR.search(dir_key.rstrip("/"))
            if match is None:
                continue
            partitions.append(
                PartitionInfo(
                    hour=match.group(1),
                    uri=f"s3://{bucket}/{dir_key}{_DATASET_LEAF}",
                )
            )
    partitions.sort(key=lambda part: part.hour)
    return partitions


def list_sealed_hours(prefix: str | os.PathLike[str]) -> list[PartitionInfo]:
    """Local ``p=<hour>/dataset.lance`` children, sorted by hour token.

    This reads directory names. It does not open the datasets.
    """
    root = Path(prefix)
    if not root.is_dir():
        raise LakeError(f"list_sealed_hours: {root} is not a directory")
    partitions: list[PartitionInfo] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _PARTITION_NAME.match(child.name)
        if match is None:
            continue
        dataset = child / _DATASET_LEAF
        if not dataset.exists():
            continue
        partitions.append(PartitionInfo(hour=match.group(1), uri=str(dataset.resolve())))
    partitions.sort(key=lambda part: part.hour)
    return partitions

"""Immutable raw-asset store shared by every ingestion client.

Every byte fetched from an external source is written once, hashed, and never
edited. Normalisation always reads from the store, never from the network, so
a result can be regenerated years later from the bytes that actually produced
it — which is the difference between a reproducible analysis and a plausible
one.

Layout, per asset:

    data/raw/<source>/<asset_id>.<ext>        the payload, verbatim
    data/raw/<source>/<asset_id>.meta.json    provenance sidecar

The sidecar records the retrieval URL, UTC retrieval time, SHA-256 of the
payload, byte count, licence and the citation the source asks for. Writing a
payload whose hash differs from an existing one raises rather than overwrites:
silent replacement of raw evidence is the failure this module exists to make
impossible.

Canonical ticket: P03_05_T001.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass

DEFAULT_ROOT = pathlib.Path("data/raw")


class RawStoreIntegrityError(RuntimeError):
    """Raised when a stored payload does not match its recorded hash."""


@dataclass(frozen=True, slots=True)
class RawAssetMetadata:
    """Provenance sidecar for one stored payload."""

    asset_id: str
    source: str
    url: str
    retrieved_at_utc: str
    sha256: str
    byte_count: int
    content_type: str
    licence: str
    licence_url: str
    citation: str
    notes: str = ""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class RawStore:
    """Content-addressed, append-only store for fetched bytes."""

    def __init__(self, root: pathlib.Path | str = DEFAULT_ROOT) -> None:
        self.root = pathlib.Path(root)

    def _paths(self, source: str, asset_id: str, ext: str) -> tuple[pathlib.Path, pathlib.Path]:
        directory = self.root / source
        return directory / f"{asset_id}{ext}", directory / f"{asset_id}.meta.json"

    def write(
        self,
        *,
        source: str,
        asset_id: str,
        payload: bytes,
        url: str,
        licence: str,
        licence_url: str,
        citation: str,
        content_type: str = "application/octet-stream",
        ext: str = ".bin",
        retrieved_at_utc: _dt.datetime | None = None,
        notes: str = "",
    ) -> RawAssetMetadata:
        """Store a payload and its provenance. Idempotent for identical bytes."""
        digest = sha256_bytes(payload)
        payload_path, meta_path = self._paths(source, asset_id, ext)

        if payload_path.exists():
            existing = sha256_bytes(payload_path.read_bytes())
            if existing != digest:
                raise RawStoreIntegrityError(
                    f"{payload_path} already exists with a different hash "
                    f"({existing[:12]}... vs {digest[:12]}...). Raw assets are "
                    "immutable; write under a new asset_id instead of overwriting."
                )
            return self.read_metadata(source, asset_id)

        retrieved = retrieved_at_utc or _dt.datetime.now(_dt.UTC)
        if retrieved.tzinfo is None:
            raise ValueError("retrieved_at_utc must be timezone-aware")

        metadata = RawAssetMetadata(
            asset_id=asset_id,
            source=source,
            url=url,
            retrieved_at_utc=retrieved.astimezone(_dt.UTC).isoformat(),
            sha256=digest,
            byte_count=len(payload),
            content_type=content_type,
            licence=licence,
            licence_url=licence_url,
            citation=citation,
            notes=notes,
        )

        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload)
        meta_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
        return metadata

    def read_metadata(self, source: str, asset_id: str) -> RawAssetMetadata:
        _, meta_path = self._paths(source, asset_id, "")
        if not meta_path.exists():
            raise FileNotFoundError(f"no stored metadata for {source}/{asset_id}")
        return RawAssetMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))

    def read(self, source: str, asset_id: str, ext: str = ".bin") -> bytes:
        """Read a payload back, verifying it against its recorded hash."""
        payload_path, _ = self._paths(source, asset_id, ext)
        if not payload_path.exists():
            raise FileNotFoundError(f"no stored payload at {payload_path}")
        payload = payload_path.read_bytes()
        metadata = self.read_metadata(source, asset_id)
        actual = sha256_bytes(payload)
        if actual != metadata.sha256:
            raise RawStoreIntegrityError(
                f"{payload_path} does not match its recorded hash: "
                f"expected {metadata.sha256[:12]}..., found {actual[:12]}...  "
                "The stored evidence has been altered on disk."
            )
        return payload

    def exists(self, source: str, asset_id: str, ext: str = ".bin") -> bool:
        payload_path, meta_path = self._paths(source, asset_id, ext)
        return payload_path.exists() and meta_path.exists()

    def manifest(self) -> list[RawAssetMetadata]:
        """Every asset in the store, for the provenance report."""
        out: list[RawAssetMetadata] = []
        if not self.root.exists():
            return out
        for meta_path in sorted(self.root.rglob("*.meta.json")):
            out.append(RawAssetMetadata(**json.loads(meta_path.read_text(encoding="utf-8"))))
        return out

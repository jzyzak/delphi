"""Deterministic content hashing and trial fingerprinting.

Two distinct hashes live here:

* ``content_hash`` / ``compute_record_hash`` — tamper-evidence. Every stored
  record is content-hashed and chained to the prior record in its stream.
* ``trial_fingerprint`` — trial identity for honest accounting. A *trial* is
  ``(spec_hash, canonical params, data_snapshot, universe_spec)``; it
  deliberately excludes code_sha / env / seeds so that silently re-running the
  same trial on a new build yields the *same* fingerprint and is detectable.

Canonicalization is the crux: the same logical content must always serialize to
the same bytes regardless of key ordering or equivalent encodings.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from core.registry.models import ReproMetadata


def _canonicalize(obj: Any) -> Any:
    """Recursively convert ``obj`` into a canonical, JSON-safe structure.

    Dict keys are coerced to strings (and later sorted by ``json.dumps``);
    datetimes are normalized to UTC ISO-8601; sets become sorted lists. Lists
    keep their order (sequence is semantically meaningful).
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_canonicalize(v) for v in obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a canonical JSON string (sorted keys, tight, UTF-8).

    Stable under key reordering and equivalent encodings, so equal content
    always produces equal bytes.
    """
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_hash(obj: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def compute_record_hash(
    *,
    stream_id: str,
    seq: int,
    record_kind: str,
    record_id: str,
    payload: dict[str, Any],
    prev_hash: str | None,
    knowledge_time: datetime,
) -> str:
    """Content hash of one stored record, chained to ``prev_hash``.

    Including ``prev_hash`` makes the per-stream chain tamper-evident: editing
    any record changes its hash and breaks every subsequent link.
    """
    body = {
        "stream_id": stream_id,
        "seq": seq,
        "record_kind": record_kind,
        "record_id": record_id,
        "payload": payload,
        "prev_hash": prev_hash,
        "knowledge_time": knowledge_time,
    }
    return content_hash(body)


def trial_fingerprint(meta: ReproMetadata) -> str:
    """Deterministic id of a TRIAL.

    ``hash(spec_hash, canonical(params), data_snapshot, universe_spec)``.
    Identical inputs → identical fingerprint (so the trials ledger in prompt 06
    counts honestly); any change to spec/params/dataset/universe → a new
    fingerprint. Deliberately excludes code_sha, env, seeds, and the dirty flag:
    re-running the *same* trial on a different build is still the same trial.
    """
    body = {
        "spec_hash": meta.spec_hash,
        "params": meta.params,
        "data_snapshot": {
            "as_of": meta.data_snapshot.as_of,
            "universe_spec": meta.data_snapshot.universe_spec,
        },
        "universe_spec": meta.data_snapshot.universe_spec,
    }
    return content_hash(body)

"""Memory: markdown entries with YAML frontmatter, embeddings index, retrieval.

Layout (inside the state store, default ``shared/``)::

    memory/
        index.md                     # human-readable table of contents (5 KB cap)
        entries/
            {id}.md                  # entry with YAML frontmatter
        .lock                        # flock for git commits / consolidation
    data/memory_index/
        active.json                  # {"model": "<slug>", "dim": 1024}
        {model_slug}/
            vectors.npy              # float32 matrix (optional numpy)
            meta.json                # {"ids": [...], "model": ..., "dim": ...}

Design rules from the spec (§4.3):

* entries are append-mostly; ``forget`` archives, never deletes from history;
* retrieval score = 0.6 * embedding + 0.3 * tag overlap + 0.1 * recency;
* conflicts are surfaced (``[conflict]``) when relations say so;
* extraction is gated on the compression role and only runs when asked;
* consolidation is additive: it only rewrites touched entries and merges
  near-duplicates, never deletes knowledge;
* the embedding index is written atomically under a different directory per
  model so a model swap never leaves a half-written index.
"""

from __future__ import annotations

import math
import re
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from canary.core import util
from canary.core.config import Config
from canary.core.embeddings import EmbeddingBackend, EmbeddingUnavailable, cosine, resolve_backend
from canary.core.models import ModelClient, ModelError
from canary.core.observability import Log

FRONTMATTER_BOUNDARY = "---"
ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")
RELATION_KINDS = ("supersedes", "contradicts", "supports")
INDEX_CAP_BYTES = 5120
RECENT_WINDOW_S = 30 * 86400.0


# ---------------------------------------------------------------------------
# entry model
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    id: str
    body: str = ""
    tags: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    source: str = ""
    importance: float = 0.5
    relations: dict[str, list[str]] = field(default_factory=dict)
    archived: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.body.strip()

    def to_markdown(self) -> str:
        fm: dict[str, Any] = {
            "id": self.id,
            "created": self.created or util.utc_stamp(),
            "updated": self.updated or util.utc_stamp(),
            "tags": list(self.tags),
            "importance": round(float(self.importance), 3),
        }
        if self.source:
            fm["source"] = self.source
        if self.relations:
            fm["relations"] = {
                kind: sorted(set(ids))
                for kind, ids in self.relations.items()
                if kind in RELATION_KINDS and ids
            }
        if self.archived:
            fm["archived"] = True
        for key, value in self.extra.items():
            fm.setdefault(key, value)
        head = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
        return f"{FRONTMATTER_BOUNDARY}\n{head}\n{FRONTMATTER_BOUNDARY}\n\n{self.body.strip()}\n"

    @classmethod
    def from_markdown(cls, entry_id: str, text: str) -> Entry:
        body = text
        fm: dict[str, Any] = {}
        if text.startswith(FRONTMATTER_BOUNDARY):
            parts = text.split(FRONTMATTER_BOUNDARY, 2)
            if len(parts) >= 3:
                try:
                    loaded = yaml.safe_load(parts[1]) or {}
                    if isinstance(loaded, dict):
                        fm = loaded
                    body = parts[2]
                except yaml.YAMLError:
                    body = text
        relations: dict[str, list[str]] = {}
        raw_rel = fm.get("relations")
        if isinstance(raw_rel, dict):
            for kind, ids in raw_rel.items():
                if kind in RELATION_KINDS and isinstance(ids, list):
                    relations[kind] = [str(i) for i in ids]
        tags = fm.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        known = {
            "id", "created", "updated", "tags", "importance",
            "source", "relations", "archived",
        }
        extra = {k: v for k, v in fm.items() if k not in known}
        try:
            importance = float(fm.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        return cls(
            id=str(fm.get("id") or entry_id),
            body=body.strip(),
            tags=[str(t) for t in tags],
            created=str(fm.get("created") or ""),
            updated=str(fm.get("updated") or ""),
            source=str(fm.get("source") or ""),
            importance=min(1.0, max(0.0, importance)),
            relations=relations,
            archived=bool(fm.get("archived")),
            extra=extra,
        )


@dataclass
class SearchHit:
    entry: Entry
    score: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "id": self.entry.id,
            "score": round(self.score, 4),
            "tags": self.entry.tags,
            "body": self.entry.text,
        }
        if self.entry.relations:
            data["relations"] = self.entry.relations
        return data


# ---------------------------------------------------------------------------
# index (embedding vectors per model)
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Persisted vectors for one embedding model, swapped in atomically."""

    def __init__(self, root: Path, backend: EmbeddingBackend, log: Log | None = None):
        self.root = Path(root)
        self.backend = backend
        self.log = log
        self.slug = util.slugify(backend.name or "unknown")
        self.dir = self.root / self.slug
        self.meta_path = self.dir / "meta.json"
        self.vectors_path = self.dir / "vectors.dat"
        self.ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._load()

    @property
    def active(self) -> bool:
        return self.backend.available and bool(self.ids)

    def _load(self) -> None:
        meta = util.read_json(self.meta_path)
        if not meta:
            return
        ids = meta.get("ids")
        if not isinstance(ids, list):
            return
        vectors: list[list[float]] = []
        try:
            raw = self.vectors_path.read_bytes()
            dim = int(meta.get("dim") or 0)
            if dim <= 0:
                return
            count = len(raw) // (4 * dim)
            floats = struct.unpack(f"<{count * dim}f", raw[: count * dim * 4])
            vectors = [list(floats[i * dim : (i + 1) * dim]) for i in range(count)]
        except (OSError, ValueError, struct.error) as exc:
            if self.log:
                self.log.warn("memory_index_load_failed", error=str(exc))
            return
        if len(vectors) != len(ids):
            return
        self.ids = [str(i) for i in ids]
        self._vectors = vectors

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        dim = len(self._vectors[0]) if self._vectors else 0
        flat = [x for vec in self._vectors for x in vec]
        util.atomic_write_bytes(self.vectors_path, struct.pack(f"<{len(flat)}f", *flat))
        util.atomic_write_json(
            self.meta_path,
            {"ids": self.ids, "model": self.backend.name, "dim": dim, "updated": util.utc_stamp()},
        )

    def update(self, entries: Iterable[Entry], *, force: bool = False) -> int:
        """(Re)embed new/changed entries. Returns number embedded."""
        if not self.backend.available:
            return 0
        wanted = [e for e in entries if not e.archived and e.text]
        if not wanted:
            return 0
        known = set(self.ids)
        todo = [e for e in wanted if force or e.id not in known]
        if not todo:
            return 0
        try:
            vectors = self.backend.embed([f"{e.id} {' '.join(e.tags)}\n{e.text}" for e in todo])
        except EmbeddingUnavailable as exc:
            if self.log:
                self.log.warn("memory_embed_failed", error=str(exc))
            return 0
        for entry, vec in zip(todo, vectors, strict=False):
            if entry.id in known:
                idx = self.ids.index(entry.id)
                self._vectors[idx] = list(vec)
            else:
                self.ids.append(entry.id)
                self._vectors.append(list(vec))
                known.add(entry.id)
        self._save()
        return len(todo)

    def remove(self, entry_ids: Iterable[str]) -> None:
        drop = set(entry_ids)
        if not drop:
            return
        keep = [(i, v) for i, v in zip(self.ids, self._vectors, strict=False) if i not in drop]
        self.ids = [i for i, _ in keep]
        self._vectors = [v for _, v in keep]
        self._save()

    def scores(self, query: str) -> dict[str, float]:
        if not self.backend.available or not self.ids:
            return {}
        try:
            vec = self.backend.embed([query])[0]
        except EmbeddingUnavailable:
            return {}
        out: dict[str, float] = {}
        for entry_id, stored in zip(self.ids, self._vectors, strict=False):
            out[entry_id] = max(0.0, cosine(vec, stored))
        return out


# ---------------------------------------------------------------------------
# memory store
# ---------------------------------------------------------------------------

class Memory:
    """Entry store + retrieval + extraction + consolidation."""

    def __init__(
        self,
        config: Config,
        log: Log | None = None,
        *,
        backend: EmbeddingBackend | None = None,
        model_client: ModelClient | None = None,
        allow_download: bool | None = None,
    ):
        self.config = config
        self.log = log
        self.dir = config.state_path / "memory"
        self.entries_dir = self.dir / "entries"
        self.index_path = self.dir / "index.md"
        self.lock_path = self.dir / ".lock"
        self._backend = backend
        self._model = model_client
        self._allow_download = allow_download
        self._index: EmbeddingIndex | None = None
        self._cache: dict[str, Entry] = {}
        self._holding_lock = False
        util.ensure_dir(self.entries_dir)

    # -- lazy dependencies ---------------------------------------------------

    @property
    def backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._backend = resolve_backend(
                self.config, self.log, allow_download=self._allow_download
            )
        return self._backend

    @property
    def index(self) -> EmbeddingIndex:
        if self._index is None:
            self.refresh()
        assert self._index is not None
        return self._index

    def refresh(self) -> None:
        """Reload the index for the currently active embedding model."""
        active_path = self.config.data_path / "memory_index" / "active.json"
        meta = util.read_json(active_path) or {}
        wanted = self.backend.name if self.backend.available else (meta.get("model") or "")
        if not wanted:
            self._index = EmbeddingIndex(
                self.config.data_path / "memory_index", self.backend, self.log
            )
            return
        self._index = EmbeddingIndex(self.config.data_path / "memory_index", self.backend, self.log)
        if meta.get("model") == wanted:
            return
        if not self.backend.available:
            return
        stored = self.config.data_path / "memory_index" / util.slugify(wanted)
        if stored.exists() and util.read_json(stored / "meta.json"):
            # A previous model's index is still on disk; rebuild lazily when
            # the caller embeds (update(force=True)) rather than blocking here.
            if self.log:
                self.log.info("memory_index_rebuild_pending", model=self.backend.name)
        util.atomic_write_json(active_path, {"model": wanted, "updated": util.utc_stamp()})

    # -- filesystem ----------------------------------------------------------

    def entry_path(self, entry_id: str) -> Path:
        return self.entries_dir / f"{entry_id}.md"

    def all_entries(self, *, include_archived: bool = False) -> list[Entry]:
        out: list[Entry] = []
        for path in sorted(self.entries_dir.glob("*.md")):
            entry = self.get(path.stem)
            if entry is None:
                continue
            if entry.archived and not include_archived:
                continue
            out.append(entry)
        return out

    def get(self, entry_id: str) -> Entry | None:
        if entry_id in self._cache:
            return self._cache[entry_id]
        path = self.entry_path(entry_id)
        if not path.exists():
            return None
        try:
            entry = Entry.from_markdown(entry_id, path.read_text(encoding="utf-8"))
        except OSError:
            return None
        self._cache[entry_id] = entry
        return entry

    def save(
        self,
        body: str,
        *,
        tags: list[str] | None = None,
        entry_id: str | None = None,
        source: str = "",
        importance: float = 0.5,
        relations: dict[str, list[str]] | None = None,
    ) -> Entry:
        body = (body or "").strip()
        if not body:
            raise ValueError("memory entry body must not be empty")
        entry_id = entry_id or util.slugify(body[:60]) or util.new_nonce()[:12]
        if not ENTRY_ID_RE.match(entry_id):
            entry_id = util.slugify(entry_id) or util.new_nonce()[:12]
        now = util.utc_stamp()
        existing = self.get(entry_id)
        if existing is not None:
            existing.body = body
            existing.updated = now
            if tags:
                existing.tags = sorted(set(existing.tags) | set(tags))
            if relations:
                for kind, ids in relations.items():
                    if kind in RELATION_KINDS:
                        merged = set(existing.relations.get(kind, [])) | set(ids)
                        existing.relations[kind] = sorted(merged)
            if source:
                existing.source = source
            entry = existing
        else:
            entry = Entry(
                id=entry_id,
                body=body,
                tags=sorted(set(tags or [])),
                created=now,
                updated=now,
                source=source,
                importance=min(1.0, max(0.0, importance)),
                relations={
                    kind: sorted(set(ids))
                    for kind, ids in (relations or {}).items()
                    if kind in RELATION_KINDS and ids
                },
            )
        util.atomic_write_text(self.entry_path(entry.id), entry.to_markdown())
        self._cache[entry.id] = entry
        if self.backend.available:
            try:
                self.index.update([entry], force=True)
            except Exception as exc:  # index must never break a save
                if self.log:
                    self.log.warn("memory_index_update_failed", error=str(exc))
        self.write_index()
        self._git_commit(
            f"state: memory save {entry.id}",
            [f"memory/entries/{entry.id}.md", "memory/index.md"],
        )
        if self.log:
            self.log.info("memory_saved", entry_id=entry.id, tags=entry.tags)
        return entry

    def forget(self, entry_id: str) -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.archived = True
        entry.updated = util.utc_stamp()
        util.atomic_write_text(self.entry_path(entry.id), entry.to_markdown())
        self._cache[entry.id] = entry
        if self._index is not None:
            self._index.remove([entry_id])
        self.write_index()
        self._git_commit(
            f"state: memory forget {entry_id}",
            [f"memory/entries/{entry_id}.md", "memory/index.md"],
        )
        if self.log:
            self.log.info("memory_forgotten", entry_id=entry_id)
        return True

    def write_index(self) -> None:
        """Regenerate index.md, capped at INDEX_CAP_BYTES by selection."""
        entries = self.all_entries()
        lines = ["# Memory index", ""]
        for entry in sorted(entries, key=lambda e: (e.importance, e.updated), reverse=True):
            tags = f"  _{', '.join(entry.tags)}_" if entry.tags else ""
            summary = entry.text.replace("\n", " ")[:120]
            lines.append(f"- [[{entry.id}]] {summary}{tags}")
        text = "\n".join(lines) + "\n"
        if len(text.encode("utf-8")) > INDEX_CAP_BYTES:
            kept = ["# Memory index", "", f"_({len(entries)} entries; showing most important)_", ""]
            size = sum(len(line.encode("utf-8")) + 1 for line in kept)
            for entry in sorted(entries, key=lambda e: (e.importance, e.updated), reverse=True):
                line = f"- [[{entry.id}]] {entry.text.replace(chr(10), ' ')[:80]}\n"
                if size + len(line.encode("utf-8")) > INDEX_CAP_BYTES:
                    break
                kept.append(line.rstrip())
                size += len(line.encode("utf-8"))
            text = "\n".join(kept) + "\n"
        util.atomic_write_text(self.index_path, text)

    # -- retrieval -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        tags: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[SearchHit]:
        k = int(k or self.config.get("memory.recall_k", 10))
        weights = self.config.get("memory.weights", {}) or {}
        w_embed = float(weights.get("embedding", 0.6))
        w_tag = float(weights.get("tags", 0.3))
        w_recency = float(weights.get("recency", 0.1))
        entries = self.all_entries(include_archived=include_archived)
        if not entries:
            return []
        by_id = {e.id: e for e in entries}
        embed_scores: dict[str, float] = {}
        if self.backend.available and query:
            try:
                embed_scores = self.index.scores(query)
            except Exception as exc:
                if self.log:
                    self.log.warn("memory_search_embed_failed", error=str(exc))
        want_tags = {t.lower() for t in (tags or [])}
        query_terms = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower())}
        now = time.time()
        hits: list[SearchHit] = []
        for entry in entries:
            text_l = entry.text.lower()
            embed = embed_scores.get(entry.id, 0.0)
            entry_tags = {t.lower() for t in entry.tags}
            tag_score = 0.0
            if want_tags:
                tag_score = len(want_tags & entry_tags) / len(want_tags)
            elif query_terms:
                body_terms = set(re.findall(r"[a-z0-9]{3,}", text_l + " " + " ".join(entry_tags)))
                overlap = len(query_terms & body_terms)
                tag_score = min(1.0, overlap / max(3, len(query_terms)))
            recency = 0.0
            try:
                age = now - util.parse_stamp(entry.updated or entry.created)
                recency = max(0.0, 1.0 - age / RECENT_WINDOW_S)
            except (ValueError, TypeError):
                recency = 0.0
            lexical = 0.0
            if query_terms:
                found = sum(1 for term in query_terms if term in text_l)
                lexical = found / len(query_terms)
            base = w_embed * embed + w_tag * tag_score + w_recency * recency
            score = base * (0.75 + 0.25 * entry.importance) + 0.15 * lexical * (1.0 - embed)
            hits.append(
                SearchHit(
                    entry=entry,
                    score=score,
                    components={"embedding": embed, "tags": tag_score, "recency": recency},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:k]
        if self.config.get("memory.relations", True):
            self._add_conflicts(top, by_id)
        return top

    def _add_conflicts(self, hits: list[SearchHit], by_id: dict[str, Entry]) -> None:
        """Flag contradicting/superseding pairs; append up to 2 extra entries."""
        present = {h.entry.id for h in hits}
        extra: list[Entry] = []
        related: set[str] = set()
        for hit in list(hits):
            rel = hit.entry.relations or {}
            for kind in ("contradicts", "supersedes"):
                for other_id in rel.get(kind, []):
                    if other_id == hit.entry.id:
                        continue
                    related.add(hit.entry.id)
                    related.add(other_id)
                    if other_id in present or other_id not in by_id or other_id in related:
                        continue
                    if any(e.id == other_id for e in extra):
                        continue
                    other = by_id[other_id]
                    other.extra.setdefault("_conflict", kind)
                    extra.append(other)
                    if len(extra) >= 2:
                        break
                if len(extra) >= 2:
                    break
        for hit in hits:
            if hit.entry.id in related:
                hit.entry.extra.setdefault("_conflict", "relation")
                hit.components["conflict"] = 1.0
        for entry in extra:
            hits.append(SearchHit(entry=entry, score=0.0, components={"conflict": 1.0}))

    # -- extraction ----------------------------------------------------------

    EXTRACT_PROMPT = (
        "You extract durable memories from an agent conversation.\n"
        "Return at most {max_n} entries as a JSON array. Each item:\n"
        '  {{"body": "one self-contained fact or preference", '
        '"tags": ["..."], "importance": 0.0-1.0}}\n'
        "Only include facts that will matter in future sessions (preferences, decisions, "
        "project facts, corrections). Never include secrets, credentials, or transient chatter. "
        "Return [] when nothing is worth remembering.\n\n"
        "Conversation:\n{conversation}\n"
    )

    def extract(
        self,
        conversation: str,
        *,
        max_entries: int | None = None,
        source: str = "",
        model_client: ModelClient | None = None,
    ) -> list[Entry]:
        if not self.config.get("memory.extract", True):
            return []
        max_entries = int(max_entries or self.config.get("memory.extract_max", 3))
        client = model_client or self._model
        if client is None:
            return []
        prompt = self.EXTRACT_PROMPT.format(
            max_n=max_entries, conversation=util.truncate(conversation, 12000)
        )
        try:
            response = client.call([{"role": "user", "content": prompt}])
        except ModelError as exc:
            if self.log:
                self.log.warn("memory_extract_failed", error=str(exc))
            return []
        return self._ingest_extracted(
            response.content or "", max_entries=max_entries, source=source
        )

    def _ingest_extracted(self, text: str, *, max_entries: int, source: str) -> list[Entry]:
        payload = _extract_json_array(text)
        if not payload:
            return []
        saved: list[Entry] = []
        dup_threshold = float(self.config.get("memory.dedupe", 0.92))
        existing = self.all_entries()
        for item in payload[:max_entries]:
            if not isinstance(item, dict):
                continue
            body = str(item.get("body") or "").strip()
            if len(body) < 8:
                continue
            if _looks_secret(body):
                continue
            duplicate = self._find_duplicate(body, existing, dup_threshold)
            if duplicate is not None:
                if self.log:
                    self.log.info("memory_extract_duplicate", entry_id=duplicate.id)
                continue
            try:
                entry = self.save(
                    body,
                    tags=[str(t) for t in item.get("tags") or []],
                    source=source or "extraction",
                    importance=float(item.get("importance", 0.5)),
                )
            except (ValueError, TypeError):
                continue
            saved.append(entry)
            existing.append(entry)
        if saved and self.log:
            self.log.info("memory_extracted", count=len(saved))
        return saved

    def _find_duplicate(self, body: str, entries: list[Entry], threshold: float) -> Entry | None:
        if not self.backend.available:
            norm = _normalize(body)
            for entry in entries:
                if _normalize(entry.text) == norm:
                    return entry
            return None
        try:
            vec = self.backend.embed([body])[0]
        except EmbeddingUnavailable:
            return None
        candidates = [e for e in entries if e.text]
        if not candidates:
            return None
        try:
            vecs = self.backend.embed([e.text for e in candidates])
        except EmbeddingUnavailable:
            return None
        for entry, other in zip(candidates, vecs, strict=False):
            if cosine(vec, other) >= threshold:
                return entry
        return None

    # -- consolidation -------------------------------------------------------

    def consolidate(
        self,
        *,
        model_client: ModelClient | None = None,
        idle_after_s: float | None = None,
    ) -> dict:
        """Merge near-duplicates and refresh touched entries. Additive only.

        Runs outside the lock for embedding; takes the lock non-blocking for a
        bounded batch of writes and defers the rest when contended.
        """
        if not self.config.get("memory.consolidate", True):
            return {"skipped": "disabled"}
        idle = float(
            idle_after_s
            if idle_after_s is not None
            else float(self.config.get("memory.idle_min", 15)) * 60.0
        )
        last = self._last_activity()
        if idle > 0 and (time.time() - last) < idle:
            return {"skipped": "not_idle", "idle_s": round(time.time() - last, 1)}
        entries = self.all_entries()
        if len(entries) < 2:
            return {"merged": 0, "considered": len(entries)}
        pairs = self._near_duplicate_pairs(entries)
        if not pairs:
            return {"merged": 0, "considered": len(entries)}
        lock = util.FileLock(self.lock_path, op="consolidate", wait=False, timeout=0)
        if not lock.acquire():
            if self.log:
                self.log.info("memory_consolidate_deferred", reason="lock_busy")
            return {"skipped": "lock_busy", "pairs": len(pairs)}
        self._holding_lock = True
        merged = 0
        try:
            limit = int(self.config.get("memory.consolidate_batch", 5))
            for keep_id, drop_id in pairs[:limit]:
                keep = self.get(keep_id)
                drop = self.get(drop_id)
                if keep is None or drop is None or drop.archived:
                    continue
                keep.body = _merge_bodies(keep.body, drop.body)
                keep.tags = sorted(set(keep.tags) | set(drop.tags))
                keep.updated = util.utc_stamp()
                keep.importance = max(keep.importance, drop.importance)
                keep.relations.setdefault("supersedes", [])
                if drop.id not in keep.relations["supersedes"]:
                    keep.relations["supersedes"].append(drop.id)
                drop.archived = True
                drop.updated = keep.updated
                util.atomic_write_text(self.entry_path(keep.id), keep.to_markdown())
                util.atomic_write_text(self.entry_path(drop.id), drop.to_markdown())
                self._cache[keep.id] = keep
                self._cache[drop.id] = drop
                merged += 1
            if merged:
                self.write_index()
                self._git_commit(
                    f"state: memory consolidate {merged}",
                    ["memory/entries", "memory/index.md"],
                )
        finally:
            self._holding_lock = False
            lock.release()
        if self.log:
            self.log.info("memory_consolidated", merged=merged, pairs=len(pairs))
        return {"merged": merged, "pairs": len(pairs), "considered": len(entries)}

    def _near_duplicate_pairs(self, entries: list[Entry]) -> list[tuple[str, str]]:
        threshold = float(self.config.get("memory.dedupe", 0.92))
        pairs: list[tuple[str, str]] = []
        if self.backend.available:
            try:
                vectors = self.backend.embed([e.text for e in entries])
                for i, (a, va) in enumerate(zip(entries, vectors, strict=False)):
                    for b, vb in zip(entries[i + 1 :], vectors[i + 1 :], strict=False):
                        if cosine(va, vb) >= threshold:
                            keep, drop = (a, b) if a.importance >= b.importance else (b, a)
                            pairs.append((keep.id, drop.id))
                return pairs
            except EmbeddingUnavailable:
                pass
        seen: dict[str, str] = {}
        for entry in entries:
            key = _normalize(entry.text)[:200]
            if key in seen:
                keep = self.get(seen[key])
                if keep is not None:
                    hi, lo = (keep, entry) if keep.importance >= entry.importance else (entry, keep)
                    pairs.append((hi.id, lo.id))
            else:
                seen[key] = entry.id
        return pairs

    def _last_activity(self) -> float:
        latest = 0.0
        for path in (self.entries_dir,):
            try:
                for child in path.glob("*.md"):
                    latest = max(latest, child.stat().st_mtime)
            except OSError:
                continue
        return latest or time.time()

    # -- git ---------------------------------------------------------------

    def _git_commit(self, message: str, paths: list[str]) -> None:
        if not self.config.get("memory.git", False):
            return
        if getattr(self, "_holding_lock", False):
            util.git_commit(self.config.state_path, message, paths, log=self.log)
            return
        lock = util.FileLock(self.lock_path, op="git", wait=False)
        if not lock.acquire():
            return
        try:
            util.git_commit(self.config.state_path, message, paths, log=self.log)
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = text[start : end + 1]
    try:
        import json

        data = json.loads(blob)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


SECRET_HINTS = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token=",
    "bearer ",
    "sk-",
    "private key",
)


def _looks_secret(body: str) -> bool:
    low = body.lower()
    return any(hint in low for hint in SECRET_HINTS)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()


def _merge_bodies(a: str, b: str) -> str:
    a = a.strip()
    b = b.strip()
    if not a:
        return b
    if not b or b in a:
        return a
    if a in b:
        return b
    return f"{a}\n\n{b}"


def tokens_to_budget(config: Config) -> int:
    return int(config.role("main").get("context_length") or 32768)


def format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No memories found."
    lines: list[str] = []
    for hit in hits:
        prefix = "[conflict] " if hit.components.get("conflict") else ""
        tags = f" [{', '.join(hit.entry.tags)}]" if hit.entry.tags else ""
        lines.append(f"- {prefix}{hit.entry.id}{tags}: {hit.entry.text}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

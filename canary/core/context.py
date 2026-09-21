"""Context engine: three-tier prompt, pruning, summarization, nudges.

The engine turns a session's raw history into the message list sent to the
model, applying the compression contract from spec §4.2:

* **stable tier** — SOUL.md + PERSONALITY.md + INSTRUCTIONS.md + a
  deterministic tool listing. Byte-identical between turns so the provider
  prefix cache keeps working.
* **context tier** — workspace files, recomputed only when they change.
* **volatile tier** — day-granularity timestamp, memory recall, injected
  messages, context-pressure nudges, and the running conversation.

Compression is two-tier: silent pruning first (never calls a model), then
LLM summarization when pruning cannot bring usage under threshold, the agent
calls ``compress``, or the provider reports context overflow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from canary.core import util
from canary.core.config import Config
from canary.core.memory import Memory, format_hits
from canary.core.models import ContextOverflow, ModelClient, ModelError
from canary.core.observability import Log
from canary.core.session import Session

PROTECTED_PATTERNS = (
    "task",
    "memory_*",
    "write",
    "edit",
    "propose",
    "revert",
    "compress",
    "evals_run",
)

TOOL_WEIGHTS = {
    "read": 0.4,
    "bash": 0.5,
    "job_spawn": 0.6,
    "sessions_list": 0.6,
    "session_read": 0.6,
    "memory_search": 0.6,
    "job_status": 0.6,
    "job_tail": 0.5,
    "job_log": 0.5,
}
DEFAULT_TOOL_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

# Suffix of the pointer left behind by _spill_message; also the marker that
# keeps the per-turn budget from spilling the same tool result twice.
_SPILL_MARK = "(read it if needed) ...]"


@dataclass
class Unit:
    """Atomic pruning unit: an assistant turn plus the tool results it caused."""

    messages: list[dict] = field(default_factory=list)
    index: int = 0
    tool_names: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    protected: bool = False
    pinned: bool = False
    pinned_by: str = ""
    reference: str = ""

    @property
    def size(self) -> int:
        return sum(len(json.dumps(m, default=str)) for m in self.messages)

    def text(self) -> str:
        parts: list[str] = []
        for msg in self.messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            for call in msg.get("tool_calls") or []:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                parts.append(json.dumps(fn.get("arguments") or "", default=str))
        return "\n".join(parts)


def is_protected(tool_name: str) -> bool:
    return any(util.glob_match(pattern, tool_name) for pattern in PROTECTED_PATTERNS)


def build_units(messages: list[dict]) -> list[Unit]:
    """Group messages into atomic units (tool calls + their results together)."""
    units: list[Unit] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        unit = Unit(index=i)
        unit.messages.append(msg)
        calls = msg.get("tool_calls") or []
        for call in calls:
            if not isinstance(call, dict):
                continue
            cid = str(call.get("id") or "")
            name = str((call.get("function") or {}).get("name") or "")
            if name:
                unit.tool_names.append(name)
            if cid:
                unit.tool_call_ids.append(cid)
        if calls:
            wanted = set(unit.tool_call_ids)
            j = i + 1
            while j < len(messages) and wanted:
                nxt = messages[j]
                if nxt.get("role") != "tool":
                    break
                if str(nxt.get("tool_call_id") or "") not in wanted:
                    break
                wanted.discard(str(nxt.get("tool_call_id") or ""))
                unit.messages.append(nxt)
                j += 1
            i = j
        else:
            i += 1
        unit.protected = (
            any(is_protected(name) for name in unit.tool_names) if unit.tool_names else False
        )
        units.append(unit)
    return units


def units_to_messages(units: list[Unit]) -> list[dict]:
    out: list[dict] = []
    for unit in units:
        out.extend(unit.messages)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@dataclass
class PruneReport:
    tokens_before: int = 0
    tokens_after: int = 0
    pruned_units: int = 0
    pruned_tools: list[str] = field(default_factory=list)
    spilled: list[str] = field(default_factory=list)
    released_pins: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "pruned_units": self.pruned_units,
            "pruned_tools": self.pruned_tools,
            "spilled": self.spilled,
            "released_pins": self.released_pins,
        }


@dataclass
class CompressReport:
    tokens_before: int = 0
    tokens_after: int = 0
    retained: list[str] = field(default_factory=list)
    trace: str = ""
    summary_path: str = ""

    def to_dict(self) -> dict:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "retained": self.retained,
            "trace": self.trace,
        }


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class ContextEngine:
    def __init__(
        self,
        config: Config,
        log: Log | None = None,
        *,
        memory: Memory | None = None,
        model_client: ModelClient | None = None,
    ):
        self.config = config
        self.log = log
        self.memory = memory
        self.model = model_client
        self.spill_dir = config.data_path / "tmp"
        self.context_length = int(
            config.role("main").get("context_length") or 32768
        )
        self.threshold = float(config.get("compression.threshold", 0.5))
        self.spill_threshold = int(config.get("compression.spill_threshold", 32768))
        # Per-turn ceiling on tool-result characters appended by one turn;
        # 0 disables it (spec: tools.turn_output_budget).
        self.turn_output_budget = int(config.get("tools.turn_output_budget", 24576))
        self.nudges = list(config.get("compression.nudge_at", [0.8, 0.9, 0.95]))
        self.recent_turns = int(config.get("pruning.recent_turns", 3))
        self.pin_max = int(config.get("pruning.pin_max", 50))
        # Provider ground truth for token estimates: the last model call's own
        # `prompt_tokens` together with the payload size it was measured against.
        self._provider_tokens = 0
        self._provider_chars = 0
        self._provider_session: str | None = None
        self._stable_cache: str | None = None
        self._workspace_sig: tuple | None = None
        self._workspace_cache: str | None = None

    # -- token accounting ----------------------------------------------------

    def chars_of(self, messages: list[dict]) -> int:
        """Serialized size of a payload, the unit every estimate is built from."""
        return len(json.dumps(messages, default=str))

    def clear_provider_usage(self) -> None:
        """Forget the provider anchor; estimates fall back to the char heuristic."""
        self._provider_tokens = 0
        self._provider_chars = 0
        self._provider_session = None

    def begin_turn(self, session_id: str) -> None:
        """Drop a foreign anchor: an anchor only describes one session's history."""
        if self._provider_session is not None and self._provider_session != session_id:
            self.clear_provider_usage()

    def note_provider_usage(
        self, prompt_tokens: int, messages: list[dict], session_id: str | None = None
    ) -> None:
        """Record what the provider actually charged for this payload.

        `prompt_tokens` covers the whole prompt the provider saw (identity, tool
        specs, injections), so an anchor is never an under-count of a payload.
        """
        if int(prompt_tokens or 0) <= 0:
            return
        self._provider_tokens = int(prompt_tokens)
        self._provider_chars = self.chars_of(messages)
        if session_id is not None:
            self._provider_session = session_id

    def count_tokens(self, messages: list[dict]) -> int:
        """Tokens for the next request: provider ground truth, else a heuristic.

        With an anchor the estimate is the last call's `prompt_tokens` plus
        `chars // 3` for content appended since it was taken (JSON-escaped
        payloads cost well over 4 chars/token, so 3 is deliberately
        pessimistic); without one it is the cold-start fallback
        `len(json.dumps) // 4`. The result is floored at that heuristic, so this
        never reports fewer tokens than the old estimate did, and capped at the
        provider's own `context_length`, which no request can exceed; past that
        point the ratio is already maximal, so the cap wins over the floor.
        """
        raw = self.chars_of(messages)
        old = max(1, raw // 4)
        estimate = old
        if self._provider_tokens:
            appended = max(0, raw - self._provider_chars)
            estimate = self._provider_tokens + (appended + 2) // 3
        return max(1, min(self.context_length, max(old, estimate)))

    @property
    def usable(self) -> int:
        return max(1, int(self.context_length * 0.9))

    def usage_ratio(self, messages: list[dict]) -> float:
        return self.count_tokens(messages) / self.usable

    # -- tier 1: stable ------------------------------------------------------

    def stable_tier(self, tools: list[dict] | None = None) -> str:
        if self._stable_cache is None:
            parts = [self._identity("SOUL.md"), self._identity("PERSONALITY.md")]
            instructions = self._identity("INSTRUCTIONS.md")
            parts.append(instructions)
            self._stable_cache = "\n\n".join(p.strip() for p in parts if p.strip())
        prompt = self._stable_cache
        if tools:
            ordered = sorted(tools, key=lambda t: str((t.get("function") or {}).get("name") or ""))
            lines = ["## Tools (deterministic order)"]
            for tool in ordered:
                fn = tool.get("function") or {}
                lines.append(f"- {fn.get('name')}: {fn.get('description') or ''}")
            prompt = f"{prompt}\n\n" + "\n".join(lines)
        return prompt

    def _identity(self, filename: str) -> str:
        path = self.config.state_path / filename
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return ""
        return self._builtin_identity(filename)

    def _builtin_identity(self, filename: str) -> str:
        from canary.core import config as config_mod

        mapping = {
            "SOUL.md": config_mod.BUILTIN_SOUL,
            "PERSONALITY.md": config_mod.BUILTIN_PERSONALITY,
            "INSTRUCTIONS.md": config_mod.BUILTIN_INSTRUCTIONS,
        }
        return mapping.get(filename, "")

    # -- tier 2: context -----------------------------------------------------

    def workspace_files(self) -> list[str]:
        names = self.config.get("context.workspace_files", []) or []
        paths: list[str] = []
        for name in names:
            path = self.config.workspace_path / str(name)
            if path.is_file():
                paths.append(str(path))
        return paths

    def context_tier(self) -> str:
        paths = self.workspace_files()
        sig: tuple = ()
        chunks: list[str] = []
        for path in paths:
            try:
                stat = Path(path).stat()
                sig = (*sig, path, stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        if self._workspace_sig == sig and self._workspace_cache is not None:
            return self._workspace_cache
        limit = int(self.config.get("context.workspace_bytes", 20000))
        budget = limit
        for path in paths:
            try:
                text = Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
            if not text.strip():
                continue
            allowed = max(0, budget)
            chunks.append(f"### {Path(path).name}\n{util.truncate(text, allowed)}")
            budget -= len(text)
            if budget <= 0:
                break
        self._workspace_sig = sig
        self._workspace_cache = "\n\n".join(chunks)
        return self._workspace_cache

    # -- tier 3: volatile ----------------------------------------------------

    def volatile(
        self,
        *,
        session: Session | None = None,
        injections: list[dict] | None = None,
        interrupted: bool = False,
        user_message: str = "",
        nudge: str = "",
        summary: str = "",
    ) -> list[dict]:
        out: list[dict] = []
        head: list[str] = [f"Today (UTC): {util.utc_day()}"]
        if interrupted:
            head.append("[system: previous turn was interrupted]")
        recall = self.recall(user_message)
        if recall:
            head.append(f"## Memory recall\n{recall}")
        if head:
            out.append({"role": "system", "content": "\n\n".join(head)})
        if summary:
            out.append({"role": "system", "content": f"[context summary]\n{summary}"})
        for item in injections or []:
            sender = item.get("from_agent") or "unknown"
            persist = " persist" if item.get("persist") else ""
            body = item.get("message", "")
            out.append(
                {
                    "role": "system",
                    "content": f"[system] Message from @{sender}{persist}: {body}",
                }
            )
        if nudge:
            out.append({"role": "system", "content": nudge})
        return out

    def recall(self, query: str, k: int | None = None) -> str:
        if not self.memory or not query:
            return ""
        try:
            hits = self.memory.search(query, k=k)
        except Exception as exc:
            if self.log:
                self.log.warn("context_recall_failed", error=str(exc))
            return ""
        if not hits:
            return ""
        return format_hits(hits)

    # -- pruning -------------------------------------------------------------

    def prune(
        self,
        messages: list[dict],
        *,
        session: Session | None = None,
        threshold: float | None = None,
        user_message: str = "",
    ) -> tuple[list[dict], PruneReport]:
        """Silent pruning per the contract. Never calls a model."""
        threshold = self.threshold if threshold is None else threshold
        report = PruneReport()
        report.tokens_before = self.count_tokens(messages)
        if self.usage_ratio(messages) <= threshold:
            report.tokens_after = report.tokens_before
            return messages, report

        units = build_units(messages)
        recent_cut = self._recent_cut(units)
        for unit in units:
            if unit.protected and unit.size > self.spill_threshold and session is not None:
                self._spill(unit, session, report)
        self._compute_pins(units, recent_cut, report)
        for unit in units:
            if unit.pinned:
                continue
            if unit.protected or unit.index >= recent_cut:
                continue
            unit.reference = "evict"
        scored = [
            (self._score(unit, user_message), unit)
            for unit in units
            if unit.reference == "evict"
        ]
        scored.sort(key=lambda item: item[0])
        for _score, unit in scored:
            if len(units_to_messages(units)) and self._ratio_of_units(units) <= threshold:
                break
            unit.reference = "pruned"
            report.pruned_units += 1
            report.pruned_tools.extend(unit.tool_names)
        kept = [unit for unit in units if unit.reference != "pruned"]
        pruned_units = [unit for unit in units if unit.reference == "pruned"]
        if pruned_units:
            note = self._prune_note(pruned_units, report)
            report.note = note
            kept_messages = units_to_messages(kept)
            insert_at = 1 if kept_messages and kept_messages[0].get("role") == "system" else 0
            note_message = {"role": "system", "content": note}
            kept_messages.insert(insert_at, note_message)
        else:
            kept_messages = messages
        report.tokens_after = self.count_tokens(kept_messages)
        if self.log and report.pruned_units:
            self.log.info(
                "context_pruned",
                units=report.pruned_units,
                tokens_before=report.tokens_before,
                tokens_after=report.tokens_after,
            )
        return kept_messages, report

    def _ratio_of_units(self, units: list[Unit]) -> float:
        live = [u for u in units if u.reference != "pruned"]
        return self.count_tokens(units_to_messages(live)) / self.usable

    def _recent_cut(self, units: list[Unit]) -> int:
        """Index of the first unit inside the protected recent window."""
        turns = 0
        for unit in reversed(units):
            for msg in unit.messages:
                if msg.get("role") == "user":
                    turns += 1
            if turns >= self.recent_turns:
                return unit.index
        return 0

    def _spill(self, unit: Unit, session: Session, report: PruneReport) -> None:
        for msg in unit.messages:
            self._spill_message(msg, session, report, min_chars=self.spill_threshold)

    def _spill_message(
        self,
        msg: dict,
        session: Session,
        report: PruneReport,
        *,
        min_chars: int = 0,
    ) -> bool:
        """Write one tool result to the spill file and keep a short pointer."""
        if not self._spillable(msg) or len(msg["content"]) <= min_chars:
            return False
        content = msg["content"]
        cid = str(msg.get("tool_call_id") or util.new_nonce()[:8])
        path = self.spill_dir / f"{session.id}_{cid}.out"
        lines = content.splitlines()
        head = "\n".join(lines[:100])
        if len(head) > 8192:
            head = head[:8192]
        pointer = (
            f"{head}\n[... {max(0, len(lines) - 100)} more lines; full output at {path} "
            f"(read it if needed) ...]"
        )
        if len(pointer) >= len(content):
            # A single huge line can make the pointer as big as the original;
            # spilling then costs a file and saves nothing, so leave it alone.
            return False
        try:
            util.atomic_write_text(path, content)
        except OSError:
            return False
        msg["content"] = pointer
        report.spilled.append(str(path))
        return True

    @staticmethod
    def _spillable(msg: dict) -> bool:
        """A tool result that has content and has not been spilled already."""
        content = msg.get("content")
        return (
            msg.get("role") == "tool"
            and isinstance(content, str)
            and content
            and _SPILL_MARK not in content
        )

    # -- per-turn tool-output budget -----------------------------------------

    def tool_chars(self, messages: list[dict]) -> int:
        """Total characters of tool results in ``messages``."""
        return sum(
            len(msg["content"])
            for msg in messages
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str)
        )

    def enforce_turn_output_budget(
        self,
        messages: list[dict],
        *,
        session: Session | None = None,
        budget: int | None = None,
    ) -> list[str]:
        """Spill the largest tool results until this turn's output fits budget.

        ``messages`` must hold only the tool results appended in the current
        turn: the budget bounds what the turn adds, not what it inherited. A
        budget <= 0 disables the check and returns without touching the payload,
        so short turns stay byte-identical. Returns the spill file paths.
        """
        limit = self.turn_output_budget if budget is None else budget
        if limit <= 0 or session is None:
            return []
        total = self.tool_chars(messages)
        if total <= limit:
            return []
        report = PruneReport()
        ranked = sorted(
            (msg for msg in messages if self._spillable(msg)),
            key=lambda msg: len(msg["content"]),
            reverse=True,
        )
        for msg in ranked:
            if total <= limit:
                break
            before = len(msg["content"])
            if not self._spill_message(msg, session, report):
                continue
            total -= before - len(msg["content"])
        return report.spilled

    def _compute_pins(
        self, units: list[Unit], recent_cut: int, report: PruneReport
    ) -> set[str]:
        """Pin tool results explicitly referenced by retained protected output."""
        refs: dict[str, str] = {}
        for unit in units:
            if not unit.protected or unit.index < recent_cut:
                continue
            text = unit.text()
            for call_id, name in zip(unit.tool_call_ids, unit.tool_names, strict=False):
                if call_id and call_id in text and name in (
                    "write", "edit", "memory_save", "memory_search",
                ):
                    refs[call_id] = unit.tool_call_ids[0] if unit.tool_call_ids else call_id
            for path_ref in re.findall(r"(?:shared|canary|tests|core|api)/[\w./-]+", text):
                refs[path_ref] = path_ref
        if not refs:
            return set()
        pinned: set[str] = set()
        order: list[str] = []
        for unit in units:
            if unit.index >= recent_cut:
                continue
            text = unit.text()
            for ref, owner in refs.items():
                if ref and ref in text:
                    unit.pinned = True
                    unit.pinned_by = owner
                    pinned.add(ref)
                    order.append(ref)
                    break
        while len(order) > self.pin_max:
            released = order.pop(0)
            for unit in units:
                if unit.pinned and unit.pinned_by == released:
                    unit.pinned = False
                    unit.reference = "released"
                    report.released_pins.append(released)
                    for msg in unit.messages:
                        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                            msg["content"] += f"\n[dependency pruned: {released}]"
        return pinned

    def _score(self, unit: Unit, user_message: str) -> float:
        weight = DEFAULT_TOOL_WEIGHT
        if unit.tool_names:
            weight = max(TOOL_WEIGHTS.get(name, DEFAULT_TOOL_WEIGHT) for name in unit.tool_names)
        recency = 1.0 / (1.0 + unit.index)
        relevance = 0.0
        if user_message:
            terms = {t for t in re.findall(r"[a-z0-9]{3,}", user_message.lower())}
            body = unit.text().lower()
            if terms:
                relevance = sum(1 for term in terms if term in body) / len(terms)
        return recency * (1.0 - weight) + relevance * weight

    def _prune_note(self, units: list[Unit], report: PruneReport) -> str:
        tools = sorted({name for unit in units for name in unit.tool_names})
        detail = f"tools: {', '.join(tools)}" if tools else "plain messages"
        note = (
            f"[system] Pruned {len(units)} older units from context ({detail}) to stay "
            f"under the compression threshold. Full history remains in history.jsonl; "
            f"tool outputs can be regenerated or read from shared/data/tmp/."
        )
        if report.released_pins:
            note += f" Released pin(s): {', '.join(report.released_pins)}."
        return note

    # -- summarization -------------------------------------------------------

    SUMMARY_PROMPT = (
        "Summarize the following conversation span for continued work. Keep:\n"
        "- decisions and their reasons\n"
        "- open tasks, file paths, ids, command lines\n"
        "- errors encountered and how they were resolved\n"
        "Drop: greetings, redundant tool chatter, restatements.\n"
        "Write compact prose plus bullet points. Do not invent facts.\n"
        "{focus_line}\nSpan:\n{span}\n"
    )

    def summarize(
        self,
        messages: list[dict],
        *,
        focus: str = "",
        model_client: ModelClient | None = None,
        session: Session | None = None,
        save_memory: bool = True,
    ) -> tuple[str, CompressReport]:
        """LLM summarization of a span. Protected outputs are kept verbatim."""
        client = model_client or self.model
        report = CompressReport()
        report.tokens_before = self.count_tokens(messages)
        if client is None:
            text = self._fallback_summary(messages)
        else:
            focus_line = f"Focus especially on: {focus}" if focus else ""
            span = util.truncate(
                "\n".join(
                    f"[{m.get('role')}] {m.get('content') or ''}"
                    for m in messages
                    if m.get("content")
                ),
                24000,
            )
            prompt = self.SUMMARY_PROMPT.format(focus_line=focus_line, span=span)
            try:
                response = client.call([{"role": "user", "content": prompt}])
                text = (response.content or "").strip() or self._fallback_summary(messages)
            except (ModelError, ContextOverflow) as exc:
                if self.log:
                    self.log.warn("context_summary_failed", error=str(exc))
                text = self._fallback_summary(messages)
        summary_message = {"role": "system", "content": f"[context summary]\n{text}"}
        report.tokens_after = self.count_tokens([summary_message])
        for msg in messages:
            if msg.get("role") == "assistant" and (msg.get("tool_calls") or []):
                for call in msg.get("tool_calls") or []:
                    name = str((call.get("function") or {}).get("name") or "")
                    if is_protected(name):
                        report.retained.append(str(call.get("id") or name))
        trace = self.spill_dir / f"summary_{util.utc_stamp()}.txt"
        try:
            util.atomic_write_text(trace, json.dumps(messages, indent=1, default=str))
            report.trace = str(trace)
        except OSError:
            report.trace = ""
        if save_memory and self.memory is not None:
            try:
                source = "context-summary"
                if session is not None:
                    source = f"context-summary:{session.id}"
                    if session.parent_session:
                        source += f":parent={session.parent_session}"
                        if session.branch_point:
                            source += f":branch={session.branch_point}"
                entry = self.memory.save(text, tags=["context-summary"], source=source)
                report.summary_path = f"memory/entries/{entry.id}.md"
            except (ValueError, OSError) as exc:
                if self.log:
                    self.log.warn("context_summary_memory_failed", error=str(exc))
        if self.log:
            self.log.info(
                "context_summarized",
                tokens_before=report.tokens_before,
                tokens_after=report.tokens_after,
            )
        return text, report

    def _fallback_summary(self, messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            first = content.splitlines()[0][:160]
            lines.append(f"- [{msg.get('role')}] {first}")
        return "Conversation span (mechanical summary):\n" + "\n".join(lines[-40:])

    # -- nudges --------------------------------------------------------------

    def nudge_for(self, ratio: float) -> str:
        triggered = [n for n in self.nudges if ratio >= float(n)]
        if not triggered:
            return ""
        level = max(triggered)
        pct = int(round(ratio * 100))
        return (
            f"[system] Context usage {pct}% (nudge at {int(level * 100)}%). "
            "Options: call compress to summarize older turns, write notes to memory "
            "and forget them here, or narrow the current task."
        )

    # -- compress tool -------------------------------------------------------

    def compress(
        self,
        messages: list[dict],
        *,
        focus: str = "",
        session: Session | None = None,
        model_client: ModelClient | None = None,
    ) -> tuple[list[dict], CompressReport]:
        """Explicit compaction: summarize everything but the recent window."""
        units = build_units(messages)
        cut = self._recent_cut(units)
        older = [u for u in units if u.index < cut]
        recent = [u for u in units if u.index >= cut]
        protected_kept = [u for u in older if u.protected]
        to_summarize = [u for u in older if not u.protected]
        if not to_summarize:
            empty = CompressReport(tokens_before=self.count_tokens(messages))
            empty.tokens_after = empty.tokens_before
            empty.retained = [name for u in protected_kept for name in u.tool_names]
            return messages, empty
        text, report = self.summarize(
            units_to_messages(to_summarize),
            focus=focus,
            model_client=model_client,
            session=session,
        )
        head = [m for m in messages[:1] if m.get("role") == "system"]
        out = list(head)
        summary_message = {"role": "system", "content": f"[context summary]\n{text}"}
        out.append(summary_message)
        for unit in protected_kept:
            out.extend(unit.messages)
        for unit in recent:
            out.extend(unit.messages)
        report.tokens_after = self.count_tokens(out)
        if self.log:
            self.log.info(
                "context_compressed",
                tokens_before=report.tokens_before,
                tokens_after=report.tokens_after,
            )
        return out, report

    # -- full build ----------------------------------------------------------

    def build(
        self,
        session: Session,
        *,
        user_message: str = "",
        tools: list[dict] | None = None,
        injections: list[dict] | None = None,
        summary: str = "",
        max_history: int | None = None,
    ) -> list[dict]:
        """Assemble the complete message list for the next model call."""
        system_parts = [self.stable_tier(tools), self.context_tier()]
        system_content = "\n\n".join(p for p in system_parts if p.strip())
        messages: list[dict] = [{"role": "system", "content": system_content}]
        history = session.history(max_history)
        history_messages = [m.to_openai() for m in history]
        interrupted = session.take_interrupted()
        messages.extend(history_messages)
        messages, _ = self.prune(messages, session=session, user_message=user_message)
        if summary:
            messages.insert(1, {"role": "system", "content": f"[context summary]\n{summary}"})
        messages.extend(
            self.volatile(
                session=session,
                injections=injections,
                interrupted=interrupted,
                user_message=user_message,
                nudge=self.nudge_for(self.usage_ratio(messages)),
            )
        )
        return messages

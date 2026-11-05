from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .config import ROOT
from .context import CommandContext
from .slots import RESULT_WORDS, TAB_WORDS, parse_ordinal, parse_tab
from .text import normalize

log = logging.getLogger(__name__)

INTENTS_PATH = ROOT / "intents.json"
SUPPORTED_VERSION = 1

_QUERY_RE = r".+?"


@dataclass(frozen=True)
class SlotDefinition:
    regex: str
    parse: Callable[[str], Any]


@dataclass(frozen=True)
class Intent:
    id: str
    action: str
    priority: int
    patterns: tuple[re.Pattern[str], ...]
    slots: Mapping[str, str]
    requires: str | None = None


@dataclass(frozen=True)
class IntentMatch:
    intent: Intent
    slots: Mapping[str, Any]


class IntentSet:
    """Декларативные интенты: шаблоны фраз, типизированные слоты, приоритеты."""

    def __init__(
        self,
        intents: Iterable[Intent] = (),
        slot_defs: Mapping[str, SlotDefinition] | None = None,
    ) -> None:
        self.intents = tuple(
            sorted(intents, key=lambda item: item.priority, reverse=True)
        )
        self._defs = dict(slot_defs or {})

    # ------------------------------------------------------------------ public
    @classmethod
    def from_data(
        cls,
        data: Any,
        site_names: Iterable[str] = (),
        volume_parser: Callable[[str], int | None] | None = None,
    ) -> IntentSet:
        if not isinstance(data, dict):
            raise ValueError("интенты должны быть JSON-объектом")
        version = data.get("version", SUPPORTED_VERSION)
        if version != SUPPORTED_VERSION:
            raise ValueError(f"неподдерживаемая версия интентов: {version}")
        raw_intents = data.get("intents")
        if not isinstance(raw_intents, list):
            raise ValueError("поле intents должно быть массивом")

        slot_defs = _slot_definitions(site_names, volume_parser)
        intents: list[Intent] = []
        ids: set[str] = set()
        for index, raw in enumerate(raw_intents):
            if not isinstance(raw, dict):
                raise ValueError(f"интент №{index + 1} должен быть объектом")
            intent_id = str(raw.get("id") or "").strip()
            if not intent_id:
                raise ValueError(f"у интента №{index + 1} нет id")
            if intent_id in ids:
                raise ValueError(f"повторяющийся id интента: {intent_id}")
            ids.add(intent_id)

            action = str(raw.get("action") or "").strip()
            if not action:
                raise ValueError(f"у интента {intent_id} нет action")

            requires = raw.get("requires")
            if requires is not None:
                requires = str(requires)
                if requires not in CommandContext.KINDS:
                    raise ValueError(
                        f"у интента {intent_id} неизвестный контекст: {requires}"
                    )

            try:
                priority = int(raw.get("priority", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"у интента {intent_id} неверный priority") from exc

            raw_slots = raw.get("slots") or {}
            if not isinstance(raw_slots, dict):
                raise ValueError(f"у интента {intent_id} поле slots должно быть объектом")
            slots: dict[str, str] = {}
            for name, value in raw_slots.items():
                slot_name = str(name)
                slot_type = str(value.get("type", "")) if isinstance(value, dict) else str(value)
                if slot_type not in slot_defs:
                    raise ValueError(
                        f"у интента {intent_id} неизвестный слот {slot_name}: {slot_type}"
                    )
                slots[slot_name] = slot_type

            raw_patterns = raw.get("patterns")
            if not isinstance(raw_patterns, list) or not raw_patterns:
                raise ValueError(f"у интента {intent_id} нет patterns")
            patterns: list[re.Pattern[str]] = []
            for template in raw_patterns:
                source = str(template or "").strip()
                if not source:
                    raise ValueError(f"у интента {intent_id} пустой шаблон")
                patterns.append(_compile_template(intent_id, source, slots, slot_defs))

            intents.append(Intent(intent_id, action, priority, tuple(patterns), slots, requires))
        return cls(intents, slot_defs)

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        site_names: Iterable[str] = (),
        volume_parser: Callable[[str], int | None] | None = None,
    ) -> IntentSet:
        path = Path(path or INTENTS_PATH)
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            log.warning("Интенты не найдены: %s", path)
            return cls()
        except OSError as exc:
            log.warning("Не удалось прочитать интенты %s: %s", path, exc)
            return cls()
        try:
            return cls.from_data(json.loads(raw), site_names, volume_parser)
        except ValueError as exc:
            log.warning("Интенты %s пропущены: %s", path, exc)
            return cls()

    def match(self, text: str, context: CommandContext | None = None) -> IntentMatch | None:
        if not text or not self.intents:
            return None
        for intent in self.intents:
            if intent.requires and (context is None or not context.matches(intent.requires)):
                continue
            for pattern in intent.patterns:
                match = pattern.fullmatch(text)
                if not match:
                    continue
                slots = _parse_slots(intent, match, self._defs)
                if slots is None:
                    continue
                return IntentMatch(intent, slots)
        return None


def _slot_definitions(
    site_names: Iterable[str],
    volume_parser: Callable[[str], int | None] | None,
) -> dict[str, SlotDefinition]:
    sites = sorted({normalize(name) for name in site_names if normalize(name)}, key=lambda item: (-len(item), item))
    if sites:
        site_re = "|".join(
            re.escape(site).replace(r"\ ", r"\s+") for site in sites
        )
    else:
        site_re = "(?!)"

    tab_words = "|".join(
        re.escape(word) for word in sorted(TAB_WORDS, key=len, reverse=True)
    )
    result_words = "|".join(
        re.escape(word) for word in sorted(RESULT_WORDS, key=len, reverse=True)
    )

    def parse_query(raw: str) -> str | None:
        value = normalize(raw)
        return value or None

    def parse_site(raw: str) -> str | None:
        value = normalize(raw)
        return value if value in sites else None

    def parse_volume(raw: str) -> int | None:
        if volume_parser is None:
            return None
        return volume_parser(raw)

    return {
        "query": SlotDefinition(_QUERY_RE, parse_query),
        # Ведущий шум перед командой: «элонорд громкость саундклауда десять».
        # Слот нужен только чтобы шаблон нашёл команду; значение не используется.
        "prefix": SlotDefinition(_QUERY_RE, parse_query),
        "site": SlotDefinition(f"(?:{site_re})", parse_site),
        "tab": SlotDefinition(rf"(?:\d{{1,2}}|{tab_words})", parse_tab),
        "ordinal": SlotDefinition(
            rf"(?:\d{{1,2}}|{result_words})", parse_ordinal
        ),
        "ordinal_word": SlotDefinition(
            rf"(?:{result_words})", lambda raw: parse_ordinal(raw, words_only=True)
        ),
        "volume": SlotDefinition(_QUERY_RE, parse_volume),
    }


def _compile_template(
    intent_id: str,
    source: str,
    slots: Mapping[str, str],
    slot_defs: Mapping[str, SlotDefinition],
) -> re.Pattern[str]:
    parts: list[str] = []
    seen: set[str] = set()
    for token in source.split():
        if token.startswith("{") and token.endswith("}"):
            name = token[1:-1]
            if not name or not re.fullmatch(r"\w+", name):
                raise ValueError(f"у интента {intent_id} неверный слот: {token}")
            if name not in slots:
                raise ValueError(
                    f"у интента {intent_id} слот {name} не описан в slots"
                )
            if name in seen:
                raise ValueError(
                    f"у интента {intent_id} слот {name} повторяется в шаблоне"
                )
            seen.add(name)
            parts.append(f"(?P<{name}>{slot_defs[slots[name]].regex})")
        else:
            if "{" in token or "}" in token:
                raise ValueError(
                    f"у интента {intent_id} слот должен быть отдельным словом: {token}"
                )
            literal = normalize(token)
            if literal:
                parts.append(re.escape(literal))
    if not parts:
        raise ValueError(f"у интента {intent_id} пустой шаблон")
    try:
        return re.compile(r"\s+".join(parts))
    except re.error as exc:
        raise ValueError(f"у интента {intent_id} неверный шаблон: {exc}") from exc


def _parse_slots(
    intent: Intent,
    match: re.Match[str],
    slot_defs: Mapping[str, SlotDefinition],
) -> dict[str, Any] | None:
    slots: dict[str, Any] = {}
    for name, raw in match.groupdict().items():
        if raw is None:
            return None
        slot_type = intent.slots.get(name)
        if slot_type is None:
            return None
        value = slot_defs[slot_type].parse(raw)
        if value is None:
            return None
        slots[name] = value
    return slots

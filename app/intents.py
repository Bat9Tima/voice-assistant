from __future__ import annotations

import itertools
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class CompiledPattern:
    """Шаблон в двух видах: точный regex и набор слов для перестановок."""

    source: str
    regex: re.Pattern[str]
    literals: tuple[str, ...]
    slots: tuple[str, ...]


@dataclass(frozen=True)
class Intent:
    id: str
    action: str
    priority: int
    patterns: tuple[CompiledPattern, ...]
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
            patterns: list[CompiledPattern] = []
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
        text = normalize(text)
        tokens = tuple(text.split())
        for intent in self.intents:
            if intent.requires and (context is None or not context.matches(intent.requires)):
                continue
            # Сначала точный порядок слов: он даёт самые надёжные слоты.
            for pattern in intent.patterns:
                match = pattern.regex.fullmatch(text)
                if not match:
                    continue
                slots = _parse_slots(intent, match, self._defs)
                if slots is None:
                    continue
                return IntentMatch(intent, slots)
            # Затем те же шаблоны, но без учёта порядка слов. Приоритет
            # интента важнее: «включи скучно в лайкнутых» должен остаться
            # в лайках, а не уйти в общий поиск SoundCloud. Внутри интента
            # выбираем шаблон с наибольшим числом ключевых слов.
            best: tuple[tuple[int, int, int, int], dict[str, Any]] | None = None
            for pattern in intent.patterns:
                found = _match_unordered(intent, pattern, tokens, self._defs)
                if found is not None and (best is None or found[0] > best[0]):
                    best = found
            if best is not None:
                return IntentMatch(intent, best[1])
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
) -> CompiledPattern:
    parts: list[str] = []
    literals: list[str] = []
    slot_names: list[str] = []
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
            slot_names.append(name)
            parts.append(f"(?P<{name}>{slot_defs[slots[name]].regex})")
        else:
            if "{" in token or "}" in token:
                raise ValueError(
                    f"у интента {intent_id} слот должен быть отдельным словом: {token}"
                )
            literal = normalize(token)
            if literal:
                literals.append(literal)
                parts.append(re.escape(literal))
    if not parts:
        raise ValueError(f"у интента {intent_id} пустой шаблон")
    try:
        regex = re.compile(r"\s+".join(parts))
    except re.error as exc:
        raise ValueError(f"у интента {intent_id} неверный шаблон: {exc}") from exc
    return CompiledPattern(source, regex, tuple(literals), tuple(slot_names))


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


# --------------------------------------------------------------------------
# Подбор без учёта порядка слов
_MAX_LITERAL_ASSIGNMENTS = 500
_FREE_SLOTS = frozenset({"query", "prefix"})


def _match_unordered(
    intent: Intent,
    pattern: CompiledPattern,
    tokens: tuple[str, ...],
    slot_defs: Mapping[str, SlotDefinition],
) -> tuple[tuple[int, int, int, int], dict[str, Any]] | None:
    """Проверяет шаблон как набор слов, а не как последовательность.

    Все литералы шаблона должны быть во фразе, а оставшиеся слова
    раскладываются по слотам. Из нескольких вариантов выбирается тот,
    где литералы стоят ближе друг к другу и чаще сохраняют порядок.
    """
    if not pattern.literals and any(
        intent.slots.get(name) in _FREE_SLOTS for name in pattern.slots
    ):
        # Шаблон из одного свободного слота совпал бы с любой фразой.
        return None
    if Counter(pattern.literals) - Counter(tokens):
        return None

    has_prefix = "prefix" in pattern.slots
    best: tuple[tuple[int, int, int, int], dict[str, Any]] | None = None
    for positions in _literal_positions(tokens, pattern.literals):
        # prefix — это шум перед командой («элонорд громкость …»).
        # Он может занимать только слова до первого ключевого слова,
        # иначе «громкость сомкала на пять» станет системной громкостью.
        prefix_count = min(positions) if has_prefix and positions else 0
        if has_prefix and prefix_count == 0:
            continue
        removed = set(positions)
        remaining = [
            token for index, token in enumerate(tokens) if index not in removed
        ]
        slots = _assign_slots(
            intent,
            pattern.slots,
            remaining,
            slot_defs,
            prefix_count if has_prefix else None,
        )
        if slots is None:
            continue
        score = (len(pattern.literals), *_order_score(positions))
        if best is None or score > best[0]:
            best = (score, slots)
    return best


def _literal_positions(
    tokens: tuple[str, ...], literals: tuple[str, ...]
) -> Iterator[tuple[int, ...]]:
    """Перебирает позиции литералов в фразе без повторов.

    Ограничение нужно только как страховка от патологической фразы
    из одинаковых слов; обычные команды дают единицы вариантов.
    """
    if not literals:
        yield ()
        return

    positions_by_token: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        positions_by_token.setdefault(token, []).append(index)

    chosen: list[int] = []
    used: set[int] = set()
    seen: set[tuple[int, ...]] = set()
    count = 0

    def walk(index: int) -> Iterator[tuple[int, ...]]:
        nonlocal count
        if count >= _MAX_LITERAL_ASSIGNMENTS:
            return
        if index == len(literals):
            key = tuple(chosen)
            if key not in seen:
                seen.add(key)
                count += 1
                yield key
            return
        for position in positions_by_token.get(literals[index], ()):
            if position in used:
                continue
            used.add(position)
            chosen.append(position)
            yield from walk(index + 1)
            chosen.pop()
            used.remove(position)

    yield from walk(0)


def _order_score(positions: tuple[int, ...]) -> tuple[int, int, int]:
    """Насколько выбранные позиции похожи на исходный порядок шаблона."""
    if not positions:
        return (0, 0, 0)
    adjacent = sum(
        1
        for left, right in zip(positions, positions[1:], strict=False)
        if right == left + 1
    )
    ordered = sum(
        1
        for left_index in range(len(positions))
        for right_index in range(left_index + 1, len(positions))
        if positions[left_index] < positions[right_index]
    )
    span = max(positions) - min(positions)
    return (adjacent, ordered, -span)


def _assign_slots(
    intent: Intent,
    slot_names: tuple[str, ...],
    remaining: list[str],
    slot_defs: Mapping[str, SlotDefinition],
    prefix_count: int | None = None,
) -> dict[str, Any] | None:
    if prefix_count is not None:
        if prefix_count <= 0 or prefix_count > len(remaining):
            return None
        prefix = _parse_slot(
            intent, "prefix", remaining[:prefix_count], slot_defs
        )
        if prefix is None:
            return None
        other_slots = tuple(name for name in slot_names if name != "prefix")
        rest = remaining[prefix_count:]
        if not other_slots:
            return prefix if not rest else None
        values = _assign_slots(intent, other_slots, rest, slot_defs)
        if values is None:
            return None
        return {**prefix, **values}
    if not slot_names:
        return {} if not remaining else None
    if len(remaining) < len(slot_names):
        return None
    if len(slot_names) == 1:
        return _parse_slot(intent, slot_names[0], remaining, slot_defs)
    for chunks in _chunks(remaining, len(slot_names)):
        # Порядок слотов в шаблоне может не совпадать с порядком слов
        # после перестановки, поэтому проверяем все назначения.
        for order in itertools.permutations(range(len(chunks))):
            values: dict[str, Any] = {}
            for name, chunk_index in zip(slot_names, order, strict=False):
                parsed = _parse_slot(
                    intent, name, chunks[chunk_index], slot_defs
                )
                if parsed is None:
                    break
                values.update(parsed)
            else:
                return values
    return None


def _parse_slot(
    intent: Intent,
    name: str,
    tokens: list[str],
    slot_defs: Mapping[str, SlotDefinition],
) -> dict[str, Any] | None:
    if not tokens:
        return None
    slot_type = intent.slots.get(name)
    if slot_type is None:
        return None
    value = slot_defs[slot_type].parse(" ".join(tokens))
    if value is None:
        return None
    return {name: value}


def _chunks(tokens: list[str], count: int) -> Iterator[tuple[list[str], ...]]:
    """Делит слова на ``count`` непустых кусков, сохраняя порядок."""
    if count == 1:
        yield (tokens,)
        return
    for cut in range(1, len(tokens) - count + 2):
        for rest in _chunks(tokens[cut:], count - 1):
            yield (tokens[:cut], *rest)

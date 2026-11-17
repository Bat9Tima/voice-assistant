"""Менеджер обновления памяти: объединение, замены и конфликты.

Резюме сессии — это снимок разговора. Если просто складывать снимки в
базу, память начинает противоречить себе: «люблю кофе» и «не люблю
кофе» лежат рядом, старая работа и новая работа считаются одним и тем
же. Менеджер превращает пункты резюме в отдельные факты и следит за
их состоянием:

* похожий факт обновляется, а не дублируется;
* противоположный по смыслу факт вытесняет старый и помечается
  конфликтом;
* незавершённая задача закрывается, когда в резюме появляется
  подходящее решение.

Факты хранятся в SQLite (``memory_facts``), поэтому переживают
перезапуск и не зависят от Chroma. Важность и уверенность считает
``app.importance``, а не модель.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from .importance import HIGH, MEDIUM, WEIGHT, classify
from .summary_schema import StructuredSummary, SummaryItem

log = logging.getLogger(__name__)

DEFAULT_FACTS_LIMIT = 6
DEFAULT_FACTS_MAX_CHARS = 900
DEFAULT_MERGE_SIMILARITY = 0.4
MAX_ACTIVE_FACTS = 500
FACTS_HEADER = (
    "Ключевые факты из прошлых разговоров (важное, не текущий разговор):\n"
    "Это проверенные факты, подтверждённые цитатами пользователя. Не "
    "пересказывай их списком и не выдумывай новые."
)
_META_FACT_PREFIXES = (
    "не ясно",
    "неясно",
    "неизвестно",
    "не известно",
    "непонятно",
    "не понятно",
    "не выяснено",
    "не установлено",
    "не уточнено",
    "не подтверждено",
    "не проверено",
    "нет данных",
    "нет информации",
)

_WORD = re.compile(r"[a-zа-я0-9]+")
_SUFFIXES = (
    "ами",
    "ями",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ах",
    "ях",
    "ом",
    "ем",
    "ам",
    "ям",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ой",
    "ей",
    "ий",
    "ый",
    "им",
    "ым",
    "ит",
    "ат",
    "ят",
    "ют",
    "ет",
    "ут",
    "ы",
    "и",
    "а",
    "я",
    "о",
    "е",
    "у",
    "ю",
)
_STOPWORDS = {
    "пользователь",
    "пользователя",
    "пользователю",
    "сообщил",
    "сообщила",
    "сказал",
    "сказала",
    "что",
    "это",
    "его",
    "ему",
    "она",
    "оно",
    "они",
    "я",
    "мне",
    "меня",
    "мой",
    "моя",
    "моё",
    "мы",
    "вы",
    "ты",
    "в",
    "во",
    "на",
    "по",
    "с",
    "со",
    "и",
    "а",
    "но",
    "не",
    "нет",
    "да",
    "был",
    "была",
    "было",
    "были",
    "есть",
    "очень",
    "сегодня",
    "вчера",
    "завтра",
    "также",
    "ещё",
    "еще",
    "уже",
    "просто",
    "себе",
    "себя",
    "свой",
    "своя",
    "свои",
    "как",
    "то",
    "всё",
    "все",
    "этот",
    "эта",
    "эти",
    "который",
    "которая",
    "которые",
}
_BUCKETS = (
    (
        "health",
        (
            "плохо",
            "болит",
            "боле",
            "приболе",
            "устал",
            "одинок",
            "тревог",
            "депресс",
            "страшн",
            "суицид",
            "навред",
            "врач",
            "больниц",
            "температур",
            "самочувств",
            "здоров",
            "выздоров",
            "нерв",
            "стресс",
            "плачу",
            "грустно",
        ),
    ),
    (
        "preference",
        (
            "люблю",
            "любит",
            "нравится",
            "предпочитаю",
            "предпочитает",
            "увлекаюсь",
            "увлекается",
            "обожаю",
            "обожает",
            "мечтаю",
            "мечтает",
            "привык",
        ),
    ),
    (
        "relationship",
        (
            "отношен",
            "отказ",
            "влюб",
            "свидан",
            "девушк",
            "парн",
            "жен",
            "муж",
            "семь",
            "мам",
            "пап",
        ),
    ),
    (
        "study",
        ("учус", "учусь", "учёб", "учеб", "универ", "курс", "школ", "экзамен"),
    ),
    (
        "work",
        ("работ", "собес", "стажиров", "должност", "начальник"),
    ),
    (
        "project",
        ("проект", "программ", "код", "задач"),
    ),
    (
        "leisure",
        (
            "игра",
            "фильм",
            "кино",
            "музык",
            "сериал",
            "гуля",
            "погод",
            "еда",
            "кофе",
            "чай",
            "книг",
            "фортнайт",
            "fortnite",
        ),
    ),
)
_POSITIVE = (
    "хорошо",
    "лучше",
    "полегчало",
    "нормально",
    "отлично",
    "выздоров",
    "справил",
    "рад",
    "счастлив",
    "помирил",
    "отдохнул",
)
_NEGATIVE = (
    "плохо",
    "хуже",
    "болит",
    "боле",
    "устал",
    "одинок",
    "тревож",
    "груст",
    "плач",
    "страшн",
)


@dataclass
class IngestResult:
    """Что менеджер сделал с одним резюме."""

    added: int = 0
    merged: int = 0
    superseded: int = 0
    resolved: int = 0
    active: int = 0


class MemoryManager:
    """Превращает структурированные резюме в аккуратные факты памяти."""

    def __init__(
        self,
        store: object,
        *,
        facts_limit: int = DEFAULT_FACTS_LIMIT,
        facts_max_chars: int = DEFAULT_FACTS_MAX_CHARS,
        merge_similarity: float = DEFAULT_MERGE_SIMILARITY,
    ) -> None:
        self.store = store
        self.facts_limit = max(0, int(facts_limit))
        self.facts_max_chars = max(0, int(facts_max_chars))
        self.merge_similarity = max(0.0, min(1.0, float(merge_similarity)))

    @classmethod
    def from_config(cls, store: object, section: dict | None = None) -> MemoryManager:
        section = section or {}
        return cls(
            store,
            facts_limit=_int(section.get("facts_limit"), DEFAULT_FACTS_LIMIT),
            facts_max_chars=_int(
                section.get("facts_max_chars"), DEFAULT_FACTS_MAX_CHARS
            ),
        )

    # ------------------------------------------------------------------ write
    def ingest(
        self,
        session_id: str,
        summary: StructuredSummary,
        *,
        created_at: datetime | None = None,
    ) -> IngestResult:
        """Добавляет пункты резюме, обновляя уже известные факты.

        Повторный вызов для той же сессии заменяет её факты, поэтому
        пересчёт резюме не плодит дубли. ``created_at`` нужен тестам и
        восстановлению старых записей.
        """
        stamp = (created_at or _now()).isoformat()
        self.store.delete_memory_facts(session_id)
        active = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        result = IngestResult()

        groups = (
            ("fact", summary.facts),
            ("decision", summary.decisions),
            ("unfinished", summary.unfinished),
        )
        for kind, items in groups:
            for item in items:
                if kind == "fact" and _is_meta_fact(item.text):
                    # «Не ясно, есть ли...» — это пробел в знаниях модели,
                    # а не сообщённый пользователем факт.
                    continue
                importance, confidence = classify(kind, item)
                key = memory_key(kind, item.text)
                merged = self._merge(
                    active, kind, key, item, importance, confidence, stamp
                )
                if merged is not None:
                    result.merged += 1
                    continue
                conflict = self._conflict(active, kind, key, item.text)
                if conflict is not None:
                    new_id = self._add(
                        session_id,
                        kind,
                        key,
                        item,
                        importance,
                        confidence,
                        stamp,
                        conflict_with=conflict["id"],
                    )
                    self.store.update_memory_fact(
                        conflict["id"],
                        status="superseded",
                        replaced_by=new_id,
                        updated_at=stamp,
                    )
                    conflict["status"] = "superseded"
                    conflict["replaced_by"] = new_id
                    result.superseded += 1
                    result.added += 1
                    self._remember(
                        active,
                        new_id,
                        session_id,
                        kind,
                        key,
                        item,
                        importance,
                        confidence,
                        stamp,
                        conflict["id"],
                    )
                    continue
                new_id = self._add(
                    session_id,
                    kind,
                    key,
                    item,
                    importance,
                    confidence,
                    stamp,
                )
                result.added += 1
                self._remember(
                    active,
                    new_id,
                    session_id,
                    kind,
                    key,
                    item,
                    importance,
                    confidence,
                    stamp,
                )

        # Решения ищем по всей активной памяти, а не только по этому
        # резюме: пересчёт сессии не должен воскрешать уже закрытую задачу.
        decisions = [
            row
            for row in active
            if row["kind"] == "decision" and row["status"] == "active"
        ]
        result.resolved = self._resolve(active, decisions, stamp)
        result.active = sum(1 for row in active if row["status"] == "active")
        if result.added or result.merged or result.superseded or result.resolved:
            log.info(
                "Память: фактов +%s, объединено %s, заменено %s, закрыто задач %s",
                result.added,
                result.merged,
                result.superseded,
                result.resolved,
            )
        return result

    # ------------------------------------------------------------------- read
    def facts(self, query: str = "", limit: int | None = None) -> list[dict]:
        """Активные важные факты, сначала самые уместные и свежие."""
        if self.facts_limit <= 0 and limit is None:
            return []
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        rows = [row for row in rows if row["importance"] in (HIGH, MEDIUM)]
        query_tokens = _token_set(query)
        rows.sort(
            key=lambda row: (
                len(query_tokens & _token_set(row["text"])),
                WEIGHT.get(row["importance"], 2),
                float(row["confidence"]),
                str(row["updated_at"]),
            ),
            reverse=True,
        )
        return rows[: max(0, int(limit if limit is not None else self.facts_limit))]

    def block(self, query: str = "") -> str:
        """Готовый блок для системного промпта или пустая строка."""
        facts = self.facts(query)
        if not facts:
            return ""
        lines = [FACTS_HEADER]
        total = len(FACTS_HEADER)
        for row in facts:
            moment = _short_date(str(row.get("updated_at") or ""))
            line = f"- {moment}{row['text']}" if moment else f"- {row['text']}"
            if self.facts_max_chars and total + len(line) + 1 > self.facts_max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines) if len(lines) > 1 else ""

    # --------------------------------------------------------------- private
    def _merge(
        self,
        active: list[dict],
        kind: str,
        key: str,
        item: SummaryItem,
        importance: str,
        confidence: float,
        stamp: str,
    ) -> dict | None:
        best: dict | None = None
        best_score = 0.0
        for row in active:
            if row["kind"] != kind or row["fact_key"] != key:
                continue
            if _opposes(row["text"], item.text):
                continue
            score = _similarity(row["text"], item.text)
            if score >= self.merge_similarity and score > best_score:
                best = row
                best_score = score
        if best is None:
            return None
        choose_new = len(_flat(item.text)) > len(_flat(best["text"]))
        text = item.text if choose_new else best["text"]
        evidence = item.evidence if choose_new else best["evidence"]
        merged_importance = _higher(best["importance"], importance)
        merged_confidence = max(float(best["confidence"]), confidence)
        self.store.update_memory_fact(
            best["id"],
            text=text,
            evidence=evidence,
            importance=merged_importance,
            confidence=merged_confidence,
            updated_at=stamp,
        )
        best.update(
            text=text,
            evidence=evidence,
            importance=merged_importance,
            confidence=merged_confidence,
            updated_at=stamp,
        )
        return best

    def _conflict(
        self,
        active: list[dict],
        kind: str,
        key: str,
        text: str,
    ) -> dict | None:
        for row in active:
            if row["kind"] != kind or row["fact_key"] != key:
                continue
            if row["status"] != "active":
                continue
            if _opposes(row["text"], text):
                return row
        return None

    def _add(
        self,
        session_id: str,
        kind: str,
        key: str,
        item: SummaryItem,
        importance: str,
        confidence: float,
        stamp: str,
        *,
        conflict_with: int | None = None,
    ) -> int:
        return self.store.add_memory_fact(
            session_id=session_id,
            kind=kind,
            fact_key=key,
            text=item.text,
            evidence=item.evidence,
            importance=importance,
            confidence=confidence,
            status="active",
            created_at=stamp,
            updated_at=stamp,
            conflict_with=conflict_with,
        )

    @staticmethod
    def _remember(
        active: list[dict],
        fact_id: int,
        session_id: str,
        kind: str,
        key: str,
        item: SummaryItem,
        importance: str,
        confidence: float,
        stamp: str,
        conflict_with: int | None = None,
    ) -> dict:
        row = {
            "id": fact_id,
            "session_id": session_id,
            "kind": kind,
            "fact_key": key,
            "text": item.text,
            "evidence": item.evidence,
            "importance": importance,
            "confidence": confidence,
            "status": "active",
            "created_at": stamp,
            "updated_at": stamp,
            "replaced_by": None,
            "conflict_with": conflict_with,
            "resolved_by": None,
        }
        active.append(row)
        return row

    def _resolve(
        self,
        active: list[dict],
        decisions: list[dict],
        stamp: str,
    ) -> int:
        resolved = 0
        for decision in decisions:
            for row in active:
                if row["kind"] != "unfinished" or row["status"] != "active":
                    continue
                if _resolves(decision["text"], row["text"]):
                    self.store.update_memory_fact(
                        row["id"],
                        status="resolved",
                        resolved_by=decision["id"],
                        updated_at=stamp,
                    )
                    row["status"] = "resolved"
                    row["resolved_by"] = decision["id"]
                    resolved += 1
        return resolved


# ------------------------------------------------------------------ helpers
def memory_key(kind: str, text: str) -> str:
    """Ключ темы: по нему факты объединяются и вытесняют друг друга."""
    flat = _flat(text)
    if kind == "fact":
        for name, markers in _BUCKETS:
            if any(marker in flat for marker in markers):
                return name
    prefix = {"fact": "fact", "decision": "decision", "unfinished": "task"}.get(
        kind, kind
    )
    tokens = _tokens(text)[:4]
    return f"{prefix}:" + " ".join(tokens) if tokens else f"{prefix}:empty"


def _similarity(left: str, right: str) -> float:
    return _jaccard(_token_set(left), _token_set(right))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _opposes(left: str, right: str) -> bool:
    left_tokens = set(_tokens(left, keep_negation=True))
    right_tokens = set(_tokens(right, keep_negation=True))
    left_negated = "не" in left_tokens or "нет" in left_tokens
    right_negated = "не" in right_tokens or "нет" in right_tokens
    if left_negated != right_negated:
        left_base = left_tokens - {"не", "нет"}
        right_base = right_tokens - {"не", "нет"}
        if _jaccard(left_base, right_base) >= 0.4:
            return True
    left_flat = _flat(left)
    right_flat = _flat(right)
    if any(word in left_flat for word in _POSITIVE) and any(
        word in right_flat for word in _NEGATIVE
    ):
        return True
    if any(word in left_flat for word in _NEGATIVE) and any(
        word in right_flat for word in _POSITIVE
    ):
        return True
    return False


def _resolves(decision: str, unfinished: str) -> bool:
    left = _token_set(decision)
    right = _token_set(unfinished)
    shared = left & right
    if len(shared) < 2:
        return False
    union = left | right
    return bool(union) and len(shared) / len(union) >= 0.25


def _higher(left: str, right: str) -> str:
    return left if WEIGHT.get(left, 0) >= WEIGHT.get(right, 0) else right


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _tokens(text: str, *, keep_negation: bool = False) -> list[str]:
    tokens: list[str] = []
    for word in _WORD.findall(_flat(text)):
        if word in _STOPWORDS and not (keep_negation and word in ("не", "нет")):
            continue
        tokens.append(_stem(word))
    return tokens


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _flat(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def _is_meta_fact(text: str) -> bool:
    """Пункт описывает пробел в знаниях, а не факт о пользователе."""
    flat = _flat(text)
    return any(flat.startswith(prefix) for prefix in _META_FACT_PREFIXES)


def _short_date(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    return moment.strftime("%d.%m.%Y: ")


def _now() -> datetime:
    return datetime.now().astimezone()


def _int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

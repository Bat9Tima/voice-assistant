"""Менеджер обновления памяти: объединение, замены и конфликты.

Резюме сессии — это снимок разговора. Если просто складывать снимки в
базу, память начинает противоречить себе: «люблю кофе» и «не люблю
кофе» лежат рядом, старая работа и новая работа считаются одним и тем
же. Менеджер превращает пункты резюме в отдельные факты и следит за
их состоянием:

* похожий факт обновляется, а не дублируется;
* явная смена состояния («выздоровел», «больше не») вытесняет старый
  факт;
* временное состояние (здоровье, настроение) может устареть само:
  более позднее противоположное утверждение заменяет его без маркера;
* постоянное предпочтение без явной смены не переписывается: спор уходит
  в ``needs_review`` до уточнения;
* незавершённая задача закрывается, когда в резюме появляется
  подходящее решение.

Длительность факта определяет ``app.durability`` по словам, а не модель.

Факты хранятся в SQLite (``memory_facts``), поэтому переживают
перезапуск и не зависят от Chroma. Важность и уверенность считает
``app.importance``, а не модель.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from .durability import (
    EVENT,
    PERMANENT,
    STATE,
    TEMPORARY,
    TRAIT,
    classify_scope,
)
from .importance import HIGH, MEDIUM, WEIGHT, classify, explicit_remember
from .summary_schema import StructuredSummary, SummaryItem

log = logging.getLogger(__name__)

DEFAULT_FACTS_LIMIT = 6
DEFAULT_FACTS_MAX_CHARS = 900
DEFAULT_MERGE_SIMILARITY = 0.4
# Временное состояние живёт сутки: «выспался» и «сейчас всё нормально»
# к следующему вечеру уже не описывают пользователя.
DEFAULT_TEMPORARY_TTL_DAYS = 1.0
# Незавершённая задача — не вечная: через две недели она либо закрыта,
# либо потеряла смысл.
DEFAULT_UNFINISHED_TTL_DAYS = 14.0
DEFAULT_UNFINISHED_LIMIT = 5
MAX_ACTIVE_FACTS = 500
NEEDS_REVIEW = "needs_review"
DISMISSED = "dismissed"
EXPIRED = "expired"
REVIEW_LIMIT = 2
FACTS_HEADER = (
    "Ключевые факты из прошлых разговоров (важное, не текущий разговор):\n"
    "Это проверенные факты, подтверждённые цитатами пользователя. Не "
    "пересказывай их списком и не выдумывай новые."
)
REVIEW_HEADER = (
    "Противоречивые сведения из прошлых разговоров (требуют уточнения):\n"
    "Не утверждай ни один вариант как факт. Если тема важна, мягко уточни "
    "у пользователя."
)
STABLE_HEADER = "Устойчивое о тебе:"
FRESH_HEADER = "Свежие ситуации:"
TOPIC_HEADER = "Релевантное запросу:"
TASKS_HEADER = "Открытые темы:"
# Явные маркеры смены состояния: «выздоровел», «больше не», «теперь».
# Только с ними система решается заменить старый факт без проверки.
_UPDATE_MARKERS = (
    "выздоров",
    "полегчало",
    "теперь",
    "уже не",
    "больше не",
    "перестал",
    "перестала",
    "бросил",
    "бросила",
    "сменил",
    "сменила",
    "переехал",
    "переехала",
    "устроился",
    "устроилась",
    "расстал",
    "рассталась",
    "помирил",
    "помирилась",
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
# Незавершённый пункт часто описывает не задачу, а пробел в разговоре:
# «пользователь не ответил…», «осталось выяснить…». Такому не место в
# памяти: через день это уже не задача, а шум.
_UNFINISHED_GAP = (
    "не ответил",
    "не ответила",
    "не рассказал",
    "не рассказала",
    "не уточнил",
    "не уточнила",
    "не объяснил",
    "не объяснила",
    "не сообщил",
    "не сообщила",
    "не назвал",
    "не назвала",
    "не подтвердил",
    "не подтвердила",
    "не продолжен",
    "не восстановлен",
    "осталось выяснить",
    "осталось непонят",
    "осталась непонят",
    "остался непонят",
    "остались непонят",
    "остались неясн",
    "что произошло",
    "что именно",
    "из-за чего",
    "по какой причин",
    "смысл и обстоятельства",
    "ассистент не уверен",
    "ассистент не стала",
    "на которую ассистент",
    "ответа пока нет",
    "пока нет",
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
    "здоров",
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
    review: int = 0
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
        temporary_ttl_days: float = DEFAULT_TEMPORARY_TTL_DAYS,
        unfinished_ttl_days: float = DEFAULT_UNFINISHED_TTL_DAYS,
        unfinished_limit: int = DEFAULT_UNFINISHED_LIMIT,
    ) -> None:
        self.store = store
        self.facts_limit = max(0, int(facts_limit))
        self.facts_max_chars = max(0, int(facts_max_chars))
        self.merge_similarity = max(0.0, min(1.0, float(merge_similarity)))
        self.temporary_ttl_days = max(0.0, float(temporary_ttl_days))
        self.unfinished_ttl_days = max(0.0, float(unfinished_ttl_days))
        self.unfinished_limit = max(0, int(unfinished_limit))

    @classmethod
    def from_config(cls, store: object, section: dict | None = None) -> MemoryManager:
        section = section or {}
        return cls(
            store,
            facts_limit=_int(section.get("facts_limit"), DEFAULT_FACTS_LIMIT),
            facts_max_chars=_int(section.get("facts_max_chars"), DEFAULT_FACTS_MAX_CHARS),
            temporary_ttl_days=_float(
                section.get("temporary_ttl_days"), DEFAULT_TEMPORARY_TTL_DAYS
            ),
            unfinished_ttl_days=_float(
                section.get("unfinished_ttl_days"), DEFAULT_UNFINISHED_TTL_DAYS
            ),
            unfinished_limit=_int(section.get("unfinished_limit"), DEFAULT_UNFINISHED_LIMIT),
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
                if kind == "unfinished" and (self.unfinished_limit <= 0 or _is_gap_task(item.text)):
                    # «Пользователь не ответил…» — не задача, а шум.
                    continue
                importance, confidence = classify(kind, item)
                scope = TRAIT if kind != "fact" else classify_scope(item.text)
                durability = TEMPORARY if scope == STATE else PERMANENT
                event_at = stamp if scope == EVENT else ""
                key = memory_key(kind, item.text)
                merged = self._merge(
                    active,
                    kind,
                    key,
                    item,
                    importance,
                    confidence,
                    stamp,
                    scope=scope,
                    event_at=event_at,
                )
                if merged is not None:
                    result.merged += 1
                    # Повторное подтверждение снимает старые сомнения.
                    self._dismiss_reviews(kind, key, stamp)
                    continue
                conflict = self._conflict(active, kind, key, item.text)
                if conflict is not None:
                    if _is_update(item.text) or _expires(conflict, durability, stamp):
                        new_id = self._add(
                            session_id,
                            kind,
                            key,
                            item,
                            importance,
                            confidence,
                            durability,
                            stamp,
                            scope=scope,
                            event_at=event_at,
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
                            durability,
                            stamp,
                            conflict["id"],
                            scope=scope,
                            event_at=event_at,
                        )
                        self._dismiss_reviews(kind, key, stamp)
                    else:
                        # Противоположное без явной смены: не выбираем
                        # сами, а прячем оба факта до уточнения.
                        new_id = self._add(
                            session_id,
                            kind,
                            key,
                            item,
                            importance,
                            confidence,
                            durability,
                            stamp,
                            scope=scope,
                            event_at=event_at,
                            status=NEEDS_REVIEW,
                            conflict_with=conflict["id"],
                        )
                        self.store.update_memory_fact(
                            conflict["id"],
                            status=NEEDS_REVIEW,
                            conflict_with=new_id,
                            updated_at=stamp,
                        )
                        conflict["status"] = NEEDS_REVIEW
                        conflict["conflict_with"] = new_id
                        result.review += 1
                        self._remember(
                            active,
                            new_id,
                            session_id,
                            kind,
                            key,
                            item,
                            importance,
                            confidence,
                            durability,
                            stamp,
                            conflict["id"],
                            status=NEEDS_REVIEW,
                            scope=scope,
                            event_at=event_at,
                        )
                    continue
                new_id = self._add(
                    session_id,
                    kind,
                    key,
                    item,
                    importance,
                    confidence,
                    durability,
                    stamp,
                    scope=scope,
                    event_at=event_at,
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
                    durability,
                    stamp,
                    scope=scope,
                    event_at=event_at,
                )
                # Новое активное утверждение важнее незакрытого спора.
                self._dismiss_reviews(kind, key, stamp)

        # Решения ищем по всей активной памяти, а не только по этому
        # резюме: пересчёт сессии не должен воскрешать уже закрытую задачу.
        decisions = [
            row for row in active if row["kind"] == "decision" and row["status"] == "active"
        ]
        result.resolved = self._resolve(active, decisions, stamp)
        result.active = sum(1 for row in active if row["status"] == "active")
        self._trim_unfinished()
        if result.added or result.merged or result.superseded or result.resolved or result.review:
            log.info(
                "Память: фактов +%s, объединено %s, обновлено %s, на проверку %s, закрыто задач %s",
                result.added,
                result.merged,
                result.superseded,
                result.review,
                result.resolved,
            )
        return result

    def promote_remembered(self) -> int:
        """Поднимает важность фактов, которые пользователь просил запомнить.

        Правило уже действует при записи, но старые записи лежат в базе с
        прежней важностью. Проход при старте чинит их без ручной правки
        SQLite.
        """
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        updated = 0
        for row in rows:
            if row["importance"] == HIGH:
                continue
            if not explicit_remember(str(row.get("evidence") or "")):
                continue
            self.store.update_memory_fact(row["id"], importance=HIGH)
            updated += 1
        if updated:
            log.info("Память: поднята важность фактов по просьбе «запомни»: %s", updated)
        return updated

    def portrait(self, limit: int = 8) -> list[dict]:
        """Компактный портрет: подтверждённые устойчивые факты о человеке.

        Это view поверх ``memory_facts``, а не отдельное хранилище. Сюда
        не попадают состояния, события и задачи: они живут в своих слоях
        промпта. Порядок детерминированный, поэтому портрет не «прыгает»
        от вопроса к вопросу.
        """
        if limit <= 0:
            return []
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        stable = [
            row
            for row in rows
            if row["kind"] in ("fact", "decision")
            and row["importance"] in (HIGH, MEDIUM)
            and _row_scope(row) == TRAIT
        ]
        stable.sort(
            key=lambda row: (
                WEIGHT.get(row["importance"], 2),
                str(row.get("updated_at") or ""),
                int(row.get("id") or 0),
                float(row["confidence"]),
            ),
            reverse=True,
        )
        return stable[:limit]

    def remembered(self, limit: int = 8) -> list[dict]:
        """Совместимость: «что ты обо мне помнишь» показывает портрет."""
        return self.portrait(limit)

    def forget(self, query: str) -> list[dict]:
        """Закрывает факты, которые пользователь просит забыть.

        Ищем по содержательным словам: «забудь кофе» находит и «любит
        кофе», и «не любит кофе». Пустой запрос ничего не закрывает:
        «забудь всё» без уточнения слишком опасно для слуха.
        """
        tokens = _forget_tokens(query)
        if not tokens:
            return []
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        matched: list[dict] = []
        for row in rows:
            text_tokens = _token_set(str(row.get("text") or ""))
            if all(any(_same_word(token, other) for other in text_tokens) for token in tokens):
                matched.append(row)
        for row in matched:
            self.store.update_memory_fact(row["id"], status=DISMISSED)
        if matched:
            log.info("Память: забыто фактов по запросу: %s", len(matched))
        return matched

    def maintain(self, *, now: datetime | None = None) -> int:
        """Чинит накопленные факты при старте.

        Правила длительности со временем меняются, поэтому старые записи
        стоит пересмотреть: ситуативное становится временным, временное
        истекает, а пробелы в разговоре перестают числиться задачами.
        Возвращает число исправленных или закрытых записей.
        """
        moment = now or _now()
        repaired = self._reclassify_temporary()
        compacted = self.compact()
        expired = self.expire_temporary(now=moment)
        pruned = self.prune_unfinished(now=moment)
        trimmed = self._trim_unfinished()
        return repaired + compacted + expired + pruned + trimmed

    def compact(self) -> int:
        """Схлопывает активные дубли внутри одной темы.

        Обычная группа — одинаковые ``kind`` и ``fact_key``: короткий
        текст считается пересказом, если все его содержательные слова уже
        есть в длинном. «Любит кофе» и «любит чай» остаются разными.

        Стиль общения — исключение: это повторяющиеся инструкции, а не
        разные факты. Они сводятся в одну строку, где сохраняются все
        найденные пожелания: прямота, мат, шутки, ирония.
        """
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            if row["kind"] not in ("fact", "decision"):
                continue
            groups.setdefault(_compact_key(row), []).append(row)
        merged = 0
        for key, group in groups.items():
            if len(group) < 2:
                continue
            if key == ("style", "style"):
                merged += self._compact_style(group)
                continue
            group.sort(
                key=lambda row: (
                    len(_token_set(str(row.get("text") or ""))),
                    len(_flat(str(row.get("text") or ""))),
                ),
                reverse=True,
            )
            kept: list[dict] = []
            for row in group:
                parent = next(
                    (
                        candidate
                        for candidate in kept
                        if not _opposes(str(candidate["text"]), str(row["text"]))
                        and _contained(str(row["text"]), str(candidate["text"]))
                    ),
                    None,
                )
                if parent is None:
                    kept.append(row)
                    continue
                self._absorb(parent, row)
                merged += 1
        merged += self._compact_unfinished(rows)
        if merged:
            log.info("Память: схлопнуто дублей фактов: %s", merged)
        return merged

    def _compact_unfinished(self, rows: list[dict]) -> int:
        """Схлопывает задачи об одном человеке или сущности.

        «Что произошло с Максимом» и «Что Максим сделал» — одна открытая
        тема, а не две. Задачи без имён не трогаем: «позвонить маме» и
        «купить продукты» должны остаться разными.
        """
        groups: list[tuple[set[str], list[dict]]] = []
        merged = 0
        for row in rows:
            if row["kind"] != "unfinished" or row["status"] != "active":
                continue
            text = str(row.get("text") or "")
            if _is_gap_task(text):
                # Пробел в разговоре — не задача, даже если рядом есть имя.
                self.store.update_memory_fact(row["id"], status=DISMISSED)
                merged += 1
                continue
            names = _unfinished_names(text)
            if not names:
                continue
            for known, items in groups:
                if known & names:
                    known |= names
                    items.append(row)
                    break
            else:
                groups.append((names, [row]))
        for _, group in groups:
            if len(group) < 2:
                continue
            group.sort(
                key=lambda row: (
                    len(_token_set(str(row.get("text") or ""))),
                    str(row.get("updated_at") or ""),
                    int(row.get("id") or 0),
                ),
                reverse=True,
            )
            parent = group[0]
            for row in group[1:]:
                if not (
                    _contained(str(row["text"]), str(parent["text"]))
                    or _similarity(str(row["text"]), str(parent["text"])) >= 0.35
                ):
                    continue
                self._absorb(parent, row)
                merged += 1
        return merged

    def _compact_style(self, group: list[dict]) -> int:
        """Сводит инструкции о манере общения в одну запись."""
        traits: set[str] = set()
        for row in group:
            traits |= _style_traits(str(row.get("text") or ""))
        if not traits:
            return 0
        group.sort(
            key=lambda row: (
                WEIGHT.get(str(row["importance"]), 0),
                str(row.get("updated_at") or ""),
                len(_flat(str(row.get("text") or ""))),
            ),
            reverse=True,
        )
        parent, rest = group[0], group[1:]
        summary = _style_summary(traits)
        confidence = max(float(row["confidence"]) for row in group)
        updated = max(str(row.get("updated_at") or "") for row in group)
        self.store.update_memory_fact(
            parent["id"],
            text=summary,
            importance=HIGH,
            confidence=confidence,
            scope=TRAIT,
            durability=PERMANENT,
            event_at="",
            updated_at=updated,
        )
        for row in rest:
            self.store.update_memory_fact(row["id"], status="superseded", replaced_by=parent["id"])
        return len(rest)

    def _absorb(
        self,
        parent: dict,
        child: dict,
        *,
        text: str | None = None,
        evidence: str | None = None,
    ) -> None:
        """Переносит содержимое дубля в оставленную запись и закрывает дубль."""
        choose_child = len(_flat(str(child["text"]))) > len(_flat(str(parent["text"])))
        merged_text = text or (str(child["text"]) if choose_child else str(parent["text"]))
        merged_evidence = evidence or (
            str(child["evidence"]) if choose_child else str(parent["evidence"])
        )
        importance = _higher(str(parent["importance"]), str(child["importance"]))
        confidence = max(float(parent["confidence"]), float(child["confidence"]))
        scope = _row_scope(child) if choose_child else _row_scope(parent)
        durability = TEMPORARY if scope == STATE else PERMANENT
        event_at = ""
        if scope == EVENT:
            event_at = str(
                (child if choose_child else parent).get("event_at")
                or parent.get("created_at")
                or child.get("created_at")
                or ""
            )
        updated = max(str(parent.get("updated_at") or ""), str(child.get("updated_at") or ""))
        self.store.update_memory_fact(
            parent["id"],
            text=merged_text,
            evidence=merged_evidence,
            importance=importance,
            confidence=confidence,
            scope=scope,
            durability=durability,
            event_at=event_at,
            updated_at=updated,
        )
        self.store.update_memory_fact(child["id"], status="superseded", replaced_by=parent["id"])

    def expire_temporary(self, *, now: datetime | None = None) -> int:
        """Закрывает временные факты старше ``temporary_ttl_days``.

        Ноль отключает срок. Задачи не трогаем: у них свой срок и своя
        логика закрытия.
        """
        if self.temporary_ttl_days <= 0:
            return 0
        moment = now or _now()
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        closed = 0
        for row in rows:
            if row["kind"] != "fact":
                continue
            # Старые записи могли остаться с durability=temporary, но без
            # заполненного scope: их всё равно надо скрыть по сроку.
            if _row_scope(row) != STATE and row["durability"] != TEMPORARY:
                continue
            age = _age_days(str(row.get("updated_at") or ""), moment)
            if age is None or age < self.temporary_ttl_days:
                continue
            self.store.update_memory_fact(row["id"], status=EXPIRED)
            closed += 1
        if closed:
            log.info("Память: истекло временных фактов: %s", closed)
        return closed

    def prune_unfinished(self, *, now: datetime | None = None) -> int:
        """Убирает из задач пробелы в разговоре и слишком старые записи."""
        moment = now or _now()
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        closed = 0
        for row in rows:
            if row["kind"] != "unfinished":
                continue
            text = str(row.get("text") or "")
            gap = _is_gap_task(text)
            age = _age_days(str(row.get("updated_at") or ""), moment)
            stale = (
                self.unfinished_ttl_days > 0 and age is not None and age >= self.unfinished_ttl_days
            )
            if not gap and not stale:
                continue
            self.store.update_memory_fact(row["id"], status=DISMISSED if gap else EXPIRED)
            closed += 1
        if closed:
            log.info("Память: закрыто незавершённых задач: %s", closed)
        return closed

    def _reclassify_temporary(self) -> int:
        """Дозаполняет scope и дату события у старых фактов.

        Раньше «сосед ужинает» считалось постоянным фактом и не уставало,
        а событие не отличалось от текущего состояния. Текст не меняется:
        достаточно поправить scope, длительность и дату. Временные записи
        не «повышаем» до черты: иначе старый шум вернулся бы в портрет.
        """
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        updated = 0
        for row in rows:
            if row["kind"] != "fact":
                continue
            text = str(row.get("text") or "")
            scope = str(row.get("scope") or "")
            durability = str(row.get("durability") or PERMANENT)
            event_at = str(row.get("event_at") or "")
            if scope in (EVENT, STATE):
                # Явно назначенный scope не пересматриваем: текст мог
                # случайно задеть маркер другого слоя («любимая группа»
                # внутри события).
                new_scope = scope
            elif durability == TEMPORARY:
                # Старые временные записи по определению описывают состояние.
                new_scope = STATE
            else:
                new_scope = classify_scope(text)
            new_event = event_at
            if new_scope == EVENT and not new_event:
                new_event = str(row.get("created_at") or row.get("updated_at") or "")
            elif new_scope != EVENT:
                new_event = ""
            new_durability = TEMPORARY if new_scope == STATE else PERMANENT
            if (new_scope, new_durability, new_event) == (scope, durability, event_at):
                continue
            self.store.update_memory_fact(
                row["id"],
                scope=new_scope,
                durability=new_durability,
                event_at=new_event,
            )
            updated += 1
        if updated:
            log.info("Память: уточнены scope и даты старых фактов: %s", updated)
        return updated

    def _trim_unfinished(self) -> int:
        """Оставляет не больше ``unfinished_limit`` открытых задач."""
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        unfinished = [row for row in rows if row["kind"] == "unfinished"]
        if len(unfinished) <= self.unfinished_limit:
            return 0
        unfinished.sort(
            key=lambda row: (
                WEIGHT.get(row["importance"], 0),
                str(row.get("updated_at") or ""),
                int(row.get("id") or 0),
            ),
            reverse=True,
        )
        closed = 0
        for row in unfinished[self.unfinished_limit :]:
            self.store.update_memory_fact(row["id"], status=DISMISSED)
            closed += 1
        if closed:
            log.info("Память: закрыто лишних незавершённых задач: %s", closed)
        return closed

    # ------------------------------------------------------------------- read
    def facts(self, query: str = "", limit: int | None = None) -> list[dict]:
        """Активные важные факты, сначала самые уместные и свежие.

        Если запрос цепляет факты, решают совпадение и важность. Если
        запрос общий («что мы обсуждали?»), первыми идут свежие: для
        последней темы важнее дата, а не заметность факта.
        """
        if self.facts_limit <= 0 and limit is None:
            return []
        rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        rows = [row for row in rows if row["importance"] in (HIGH, MEDIUM)]
        query_tokens = _token_set(query)
        overlaps = {int(row["id"]): len(query_tokens & _token_set(row["text"])) for row in rows}
        if any(overlaps.values()):
            rows.sort(
                key=lambda row: (
                    overlaps[int(row["id"])],
                    WEIGHT.get(row["importance"], 2),
                    str(row["updated_at"]),
                    float(row["confidence"]),
                ),
                reverse=True,
            )
        else:
            # Общий вопрос не должен вытесняться свежим шумом: сначала
            # устойчивые черты, потом состояния и события.
            rows.sort(
                key=lambda row: (
                    _row_scope(row) == TRAIT and row["kind"] != "unfinished",
                    str(row["updated_at"]),
                    WEIGHT.get(row["importance"], 2),
                    float(row["confidence"]),
                ),
                reverse=True,
            )
        return rows[: max(0, int(limit if limit is not None else self.facts_limit))]

    def reviews(self, limit: int = REVIEW_LIMIT) -> list[tuple[dict | None, dict]]:
        """Пары конфликтующих фактов, которые ждут уточнения.

        Возвращает ``(старый, новый)``. Старый может быть ``None``, если
        парная запись уже удалена. Показываем только важное: спор о погоде
        не стоит вопроса пользователю.
        """
        if limit <= 0:
            return []
        rows = self.store.memory_facts(status=NEEDS_REVIEW, limit=MAX_ACTIVE_FACTS)
        pairs: list[tuple[dict | None, dict]] = []
        seen: set[tuple[int, int]] = set()
        for row in rows:
            if row["importance"] not in (HIGH, MEDIUM):
                continue
            other_id = row.get("conflict_with")
            other = self.store.memory_fact(int(other_id)) if other_id else None
            if other is not None and other["status"] != NEEDS_REVIEW:
                # Пара уже закрыта обновлением или подтверждением.
                continue
            left = int(other_id) if other_id else int(row["id"])
            key = (min(int(row["id"]), left), max(int(row["id"]), left))
            if key in seen:
                continue
            seen.add(key)
            if other is None:
                pairs.append((None, row))
            elif int(row["id"]) < int(other["id"]):
                pairs.append((row, other))
            else:
                pairs.append((other, row))
            if len(pairs) >= int(limit):
                break
        return pairs

    def block(self, query: str = "", *, related: list[dict] | None = None) -> str:
        """Готовый блок для системного промпта или пустая строка.

        Внутри фактов три слоя: устойчивое (постоянные черты и решения),
        свежее (временные состояния) и найденное по смыслу. Задачи не
        выдаются за факты: для них отдельный короткий список.

        ``related`` — смысловые находки по запросу. Совпадение по словам
        надёжнее близости векторов, поэтому при точном совпадении находки
        дополняют список, а при перефразировке — идут первыми. Дубли
        отбрасываются по id.
        """
        base = self.facts(query)
        chosen = self._with_related(base, related, query)
        base_ids = {int(row["id"]) for row in base if row.get("id") is not None}
        stable: list[dict] = []
        fresh: list[dict] = []
        tasks: list[dict] = []
        topic: list[dict] = []
        for row in chosen:
            if int(row["id"]) not in base_ids:
                topic.append(row)
            elif row["kind"] == "unfinished":
                tasks.append(row)
            elif _row_scope(row) in (STATE, EVENT):
                fresh.append(row)
            else:
                stable.append(row)
        reviews = self.reviews()
        if not (stable or fresh or tasks or topic) and not reviews:
            return ""
        lines: list[str] = [FACTS_HEADER]
        total = len(FACTS_HEADER) + 1
        for header, rows in (
            (STABLE_HEADER, stable),
            (FRESH_HEADER, fresh),
            (TASKS_HEADER, tasks),
            (TOPIC_HEADER, topic),
        ):
            if not rows:
                continue
            if self.facts_max_chars and total + len(header) + 1 > self.facts_max_chars:
                break
            body: list[str] = []
            used = total + len(header) + 1
            for row in rows:
                line = _fact_line(row)
                if self.facts_max_chars and used + len(line) + 1 > self.facts_max_chars:
                    break
                body.append(line)
                used += len(line) + 1
            if body:
                lines.extend([header, *body])
                total = used
        if reviews:
            body = []
            used = total + len(REVIEW_HEADER) + 1
            for old, new in reviews:
                if old is None:
                    line = f"- Требует уточнения: {new['text']}"
                else:
                    previous = str(old["text"]).rstrip(" .")
                    line = f"- Было: {previous}. Сейчас: {new['text']}"
                if self.facts_max_chars and used + len(line) + 1 > self.facts_max_chars:
                    break
                body.append(line)
                used += len(line) + 1
            if body:
                lines.extend([REVIEW_HEADER, *body])
        return "\n".join(lines) if len(lines) > 1 else ""

    # --------------------------------------------------------------- private
    def _with_related(
        self,
        facts: list[dict],
        related: list[dict] | None,
        query: str,
    ) -> list[dict]:
        """Добавляет смысловые находки, не вытесняя точные совпадения."""
        limit = max(0, self.facts_limit)
        if not related or limit <= 0:
            return facts[:limit]
        query_tokens = _token_set(query)
        exact = bool(query_tokens) and any(
            query_tokens & _token_set(str(row.get("text") or "")) for row in facts
        )
        seen = {int(row["id"]) for row in facts if row.get("id") is not None}
        extra: list[dict] = []
        for row in related:
            if row.get("id") is None:
                continue
            fact_id = int(row["id"])
            if fact_id in seen:
                continue
            if row.get("importance") not in (HIGH, MEDIUM):
                continue
            if str(row.get("status") or "active") != "active":
                continue
            seen.add(fact_id)
            extra.append(row)
        merged = facts + extra if exact else extra + facts
        return merged[:limit]

    def _dismiss_reviews(
        self,
        kind: str,
        key: str,
        stamp: str,
    ) -> None:
        """Закрывает незавершённые споры по теме: появилось новое знание."""
        rows = self.store.memory_facts(status=NEEDS_REVIEW, limit=MAX_ACTIVE_FACTS)
        for row in rows:
            if row["kind"] == kind and row["fact_key"] == key:
                self.store.update_memory_fact(row["id"], status=DISMISSED, updated_at=stamp)

    def _merge(
        self,
        active: list[dict],
        kind: str,
        key: str,
        item: SummaryItem,
        importance: str,
        confidence: float,
        stamp: str,
        *,
        scope: str = TRAIT,
        event_at: str = "",
    ) -> dict | None:
        best: dict | None = None
        best_score = 0.0
        for row in active:
            if row["kind"] != kind or row["fact_key"] != key:
                continue
            if row["status"] != "active":
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
        merged_scope = scope if choose_new else _row_scope(best)
        merged_durability = TEMPORARY if merged_scope == STATE else PERMANENT
        merged_event = ""
        if merged_scope == EVENT:
            dates = [value for value in (best.get("event_at"), event_at) if value]
            merged_event = min(dates) if dates else ""
        self.store.update_memory_fact(
            best["id"],
            text=text,
            evidence=evidence,
            importance=merged_importance,
            confidence=merged_confidence,
            scope=merged_scope,
            durability=merged_durability,
            event_at=merged_event,
            updated_at=stamp,
        )
        best.update(
            text=text,
            evidence=evidence,
            importance=merged_importance,
            confidence=merged_confidence,
            scope=merged_scope,
            durability=merged_durability,
            event_at=merged_event,
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
        durability: str,
        stamp: str,
        *,
        scope: str = TRAIT,
        event_at: str = "",
        status: str = "active",
        conflict_with: int | None = None,
    ) -> int:
        return self.store.add_memory_fact(
            session_id=session_id,
            kind=kind,
            scope=scope,
            durability=durability,
            event_at=event_at,
            fact_key=key,
            text=item.text,
            evidence=item.evidence,
            importance=importance,
            confidence=confidence,
            status=status,
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
        durability: str,
        stamp: str,
        conflict_with: int | None = None,
        status: str = "active",
        *,
        scope: str = TRAIT,
        event_at: str = "",
    ) -> dict:
        row = {
            "id": fact_id,
            "session_id": session_id,
            "kind": kind,
            "scope": scope,
            "durability": durability,
            "event_at": event_at,
            "fact_key": key,
            "text": item.text,
            "evidence": item.evidence,
            "importance": importance,
            "confidence": confidence,
            "status": status,
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
    prefix = {"fact": "fact", "decision": "decision", "unfinished": "task"}.get(kind, kind)
    tokens = _tokens(text)[:4]
    return f"{prefix}:" + " ".join(tokens) if tokens else f"{prefix}:empty"


def _similarity(left: str, right: str) -> float:
    return _jaccard(_token_set(left), _token_set(right))


def _contained(short: str, long: str) -> bool:
    """Все содержательные слова короткого текста есть в длинном."""
    left = _token_set(short) - _GENERIC_MERGE
    right = _token_set(long) - _GENERIC_MERGE
    return bool(left) and left <= right


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _is_update(text: str) -> bool:
    """Есть ли явный маркер смены состояния, а не просто отрицание."""
    flat = _flat(text)
    return any(marker in flat for marker in _UPDATE_MARKERS)


def _expires(conflict: dict, durability: str, stamp: str) -> bool:
    """Можно ли заменить старое состояние более поздним утверждением.

    Так устроены только временные состояния: здоровье и настроение
    устаревают сами. Постоянное предпочтение без явного маркера не
    переписывается — ему нужна проверка. Факты из одного резюме
    одновременны, поэтому спор между ними тоже уходит на проверку.
    """
    if durability != TEMPORARY or conflict.get("durability") != TEMPORARY:
        return False
    return _is_later(str(conflict.get("updated_at") or ""), stamp)


def _is_later(previous: str, current: str) -> bool:
    try:
        return datetime.fromisoformat(current) > datetime.fromisoformat(previous)
    except (TypeError, ValueError):
        return False


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


# Слова, которые не делают факт другим: «очень любит» и «любит» — одно
# и то же. Нужны при проверке вложенности текстов при схлопывании дублей.
_GENERIC_MERGE = frozenset(
    _tokens(
        "очень сам также еще уже просто свой который этот весь всего "
        "любит любить нравится предпочитает обожает увлекается мечтает привык"
    )
)


def _is_meta_fact(text: str) -> bool:
    """Пункт описывает пробел в знаниях, а не факт о пользователе."""
    flat = _flat(text)
    return any(flat.startswith(prefix) for prefix in _META_FACT_PREFIXES)


def _is_gap_task(text: str) -> bool:
    """Незавершённый пункт — это не задача, а неотвеченный вопрос."""
    flat = _flat(text)
    if flat.startswith(("неясно", "непонятно", "неизвестно")):
        return True
    return any(marker in flat for marker in _UNFINISHED_GAP)


def _forget_tokens(text: str) -> list[str]:
    """Содержательные слова запроса «забудь …» без служебной обвязки."""
    tokens = []
    for token in _tokens(text):
        if token.startswith(("забуд", "сотр", "удал", "пожалуйст")):
            continue
        if token in {"про", "об", "о", "вс"}:
            continue
        tokens.append(token)
    return tokens


def _same_word(left: str, right: str) -> bool:
    """Слова совпадают с точностью до грубого стемминга."""
    if left == right:
        return True
    if len(left) < 3 or len(right) < 3:
        return False
    return left[:3] == right[:3]


# Стиль общения — не набор разных фактов, а одна повторяемая просьба.
# Важно не принять за неё жалобу («расстроен, что не хочет материться»).
_STYLE_BAD = ("расстро", "не хочет", "отказ")
_STYLE_INTENT = frozenset(
    {
        "предпочита",
        "прос",
        "просьб",
        "общаться",
        "общени",
        "ответ",
        "стил",
        "использовать",
        "чащ",
        "продолжать",
    }
)


def _style_traits(text: str) -> set[str]:
    """Какие пожелания к манере общения есть в тексте."""
    flat = _flat(text)
    if any(marker in flat for marker in _STYLE_BAD):
        return set()
    tokens = _token_set(text)
    intent = any(
        token in _STYLE_INTENT or token.startswith(("предпочита", "прос", "общ"))
        for token in tokens
    )
    if not intent:
        return set()
    traits: set[str] = set()
    if tokens & {"прям", "свойск", "разговорн"}:
        traits.add("direct")
    if tokens & {"мат", "брань", "бран", "ругательств"} or any(
        token.startswith(("материт", "ругатель")) for token in tokens
    ):
        traits.add("mat")
    if any(token.startswith("шут") for token in tokens):
        traits.add("humor")
    if any(token.startswith("ирони") for token in tokens):
        traits.add("irony")
    return traits


def _compact_key(row: dict) -> tuple[str, str]:
    """Ключ группы при схлопывании: стиль общения общий для фактов и решений."""
    kind = str(row.get("kind") or "")
    if kind in ("fact", "decision"):
        if _style_traits(str(row.get("text") or "")):
            return ("style", "style")
    return (kind, str(row.get("fact_key") or ""))


def _style_summary(traits: set[str]) -> str:
    """Одна строка из набора пожеланий к стилю общения."""
    objects = []
    if "mat" in traits:
        objects.append("уместный мат")
    if "humor" in traits:
        objects.append("шутки")
    if "irony" in traits:
        objects.append("иронию")
    if "direct" in traits:
        head = "Пользователь предпочитает прямой, разговорный стиль общения"
        if not objects:
            return head + "."
        nominative = []
        if "mat" in traits:
            nominative.append("мат")
        if "humor" in traits:
            nominative.append("шутки")
        if "irony" in traits:
            nominative.append("ирония")
        return f"{head}; уместны {_join(nominative)}."
    if not objects:
        return ""
    return f"Пользователь просит в общении {_join(objects)}."


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " и " + items[-1]


def _age_days(value: str, now: datetime) -> float | None:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=now.tzinfo)
    return (now - moment).total_seconds() / 86400


def _row_scope(row: dict) -> str:
    """Scope записи с поправкой на старые строки без заполненного поля."""
    scope = str(row.get("scope") or "")
    text = str(row.get("text") or "")
    if scope == EVENT:
        return EVENT
    if scope == STATE or row.get("durability") == TEMPORARY:
        return STATE
    if scope == TRAIT:
        # Старая база по умолчанию хранит trait. Событие, записанное до
        # появления scope, всё равно должно иметь дату и не звучать как
        # «сейчас», поэтому проверяем текст.
        return EVENT if classify_scope(text) == EVENT else TRAIT
    return classify_scope(text)


def _fact_line(row: dict) -> str:
    """Строка факта: у события дата в прошлом, у состояния — свежая."""
    text = str(row.get("text") or "")
    scope = _row_scope(row)
    if scope == EVENT:
        moment = _short_date(
            str(row.get("event_at") or row.get("created_at") or row.get("updated_at") or "")
        )
        return f"- {moment} (было): {text}" if moment else f"- (было): {text}"
    moment = _short_date(str(row.get("updated_at") or ""))
    return f"- {moment}: {text}" if moment else f"- {text}"


def _short_date(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    return moment.strftime("%d.%m.%Y")


# Слова, с которых начинается фраза, а не имя собственное.
_NON_NAME_STARTS = frozenset(
    {
        "что",
        "кто",
        "как",
        "почему",
        "зачем",
        "где",
        "когда",
        "куда",
        "откуда",
        "позвонить",
        "позвони",
        "купить",
        "купи",
        "закончить",
        "закончи",
        "проверить",
        "проверь",
        "спросить",
        "спроси",
        "узнать",
        "узнай",
        "написать",
        "напиши",
        "сделать",
        "сделай",
        "выяснить",
        "выясни",
        "поговорить",
        "поговори",
        "сходить",
        "сходи",
        "поехать",
        "поезжай",
        "решить",
        "реши",
        "подготовить",
        "подготовь",
        "посмотреть",
        "посмотри",
        "прочитать",
        "прочитай",
        "найти",
        "найди",
        "пользователь",
        "пользователю",
        "ассистент",
        "луна",
        "сегодня",
        "вчера",
        "завтра",
    }
)
_NAME_SUFFIXES = ("ами", "ями", "ом", "ем", "а", "я", "у", "ю", "е", "ы", "и")


def _unfinished_names(text: str) -> set[str]:
    """Имена в задаче: по ним задачи об одном человеке схлопываются."""
    names: set[str] = set()
    for match in re.finditer(r"[А-ЯЁ][а-яё]+", text or ""):
        word = match.group(0)
        low = word.lower().replace("ё", "е")
        if low in _NON_NAME_STARTS:
            continue
        for suffix in _NAME_SUFFIXES:
            if len(low) > len(suffix) + 3 and low.endswith(suffix):
                low = low[: -len(suffix)]
                break
        names.add(low)
    return names


def _now() -> datetime:
    return datetime.now().astimezone()


def _int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

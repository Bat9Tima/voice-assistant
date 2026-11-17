"""Правила важности для фактов памяти.

Классификация намеренно детерминированная: модель уже извлекла факт и
подтвердила его цитатой, а важность решают простые правила. Это дешевле
и стабильнее отдельного LLM-вызова, а спорные случаи всегда можно
поправить в одном месте.

Уровни:

* ``high`` — здоровье и эмоции, устойчивые предпочтения, отношения,
  явные решения и вопросы безопасности;
* ``medium`` — учёба, работа, проекты, планы и обычные незавершённые
  задачи;
* ``low`` — бытовые темы и болтовня. Такие факты хранятся для истории,
  но в блок памяти не попадают.
"""

from __future__ import annotations

from .summary_schema import SummaryItem

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
WEIGHT = {HIGH: 3, MEDIUM: 2, LOW: 1}

# Порядок важен: здоровье и предпочтения checked раньше бытовых тем.
_HEALTH = (
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
    "хуже",
    "лучше",
    "полегчало",
    "нерв",
    "стресс",
    "плачу",
    "грустно",
)
_PREFERENCE = (
    "люблю",
    "любит",
    "нравится",
    "не люблю",
    "предпочитаю",
    "предпочитает",
    "увлекаюсь",
    "увлекается",
    "обожаю",
    "обожает",
    "мечтаю",
    "мечтает",
    "привык",
)
_RELATION = (
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
)
_STUDY_WORK = (
    "работ",
    "учус",
    "учусь",
    "учёб",
    "учеб",
    "универ",
    "проект",
    "курс",
    "программ",
    "код",
    "собес",
    "стажиров",
    "школ",
    "экзамен",
    "задач",
    "планиру",
    "собираю",
    "начал",
    "закончил",
)
_LEISURE = (
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
)
_DECISION_STRONG = ("решил", "договор", "выбрал", "принял", "точно")
_SPECULATION = ("наверное", "возможно", "кажется", "может быть", "вроде")


def classify(kind: str, item: SummaryItem) -> tuple[str, float]:
    """Возвращает ``(важность, уверенность)`` для одного пункта резюме."""
    text = _flat(item.text)
    evidence = _flat(item.evidence)
    confidence = _confidence(evidence, text)

    if kind == "decision":
        return (HIGH if _has(text, _DECISION_STRONG) else MEDIUM), confidence
    if kind == "unfinished":
        return (HIGH if _has(text, _HEALTH) else MEDIUM), confidence
    if _has(text, _HEALTH):
        return HIGH, confidence
    if _has(text, _PREFERENCE):
        return HIGH, confidence
    if _has(text, _RELATION):
        return HIGH, confidence
    if _has(text, _STUDY_WORK):
        return MEDIUM, confidence
    if _has(text, _LEISURE):
        return LOW, confidence
    return MEDIUM, confidence


def _confidence(evidence: str, text: str) -> float:
    """Чем длиннее и личнее подтверждение, тем выше уверенность."""
    words = evidence.split()
    if len(words) >= 7:
        value = 0.95
    elif len(words) >= 4:
        value = 0.85
    elif len(words) >= 2:
        value = 0.75
    else:
        value = 0.6
    first_person = ("я ", "мне ", "меня ", "мой ", "моя ", "моё ", "мы ")
    if any(marker in f"{evidence} " for marker in first_person):
        value += 0.05
    if _has(text, _SPECULATION):
        value -= 0.15
    return round(min(1.0, max(0.3, value)), 2)


def _flat(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def _has(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)

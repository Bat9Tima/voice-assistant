from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ROOT
from .text import normalize

log = logging.getLogger(__name__)

DICTIONARY_PATH = ROOT / "dictionary.json"
SUPPORTED_VERSION = 1


@dataclass(frozen=True)
class Alias:
    text: str
    replace: bool = True


@dataclass(frozen=True)
class Term:
    id: str
    canonical: str
    kind: str
    aliases: tuple[Alias, ...]


@dataclass(frozen=True)
class TermMatch:
    term: Term
    alias: Alias
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.alias.text


class Dictionary:
    """Словарь предметной области: варианты названий и их каноническая форма.

    Термины ищутся по границам слов, самый длинный вариант имеет приоритет.
    Замена выполняется только для алиасов с replace=true, поэтому неоднозначные
    слова можно держать в словаре для будущих интентов, не меняя текст команды.
    """

    def __init__(self, terms: Iterable[Term] = ()) -> None:
        self.terms = tuple(terms)
        self._by_alias: dict[str, tuple[Term, Alias]] = {}
        for term in self.terms:
            for alias in term.aliases:
                existing = self._by_alias.get(alias.text)
                if existing is not None:
                    if existing[0].canonical != term.canonical:
                        raise ValueError(
                            f"алиас «{alias.text}» уже ведёт к «{existing[0].canonical}»"
                        )
                    continue
                self._by_alias[alias.text] = (term, alias)

        self._find_re = self._compile(self._by_alias)
        replacements = {
            text: term.canonical
            for text, (term, alias) in self._by_alias.items()
            if alias.replace and text != term.canonical
        }
        self._replace_re = self._compile(replacements) if replacements else None
        self._replacements = replacements

    # ------------------------------------------------------------------ public
    @classmethod
    def from_data(cls, data: Any) -> Dictionary:
        if not isinstance(data, dict):
            raise ValueError("словарь должен быть JSON-объектом")
        version = data.get("version", SUPPORTED_VERSION)
        if version != SUPPORTED_VERSION:
            raise ValueError(f"неподдерживаемая версия словаря: {version}")
        raw_terms = data.get("terms")
        if not isinstance(raw_terms, list):
            raise ValueError("поле terms должно быть массивом")

        terms: list[Term] = []
        ids: set[str] = set()
        for index, raw in enumerate(raw_terms):
            if not isinstance(raw, dict):
                raise ValueError(f"термин №{index + 1} должен быть объектом")
            term_id = str(raw.get("id") or "").strip()
            if not term_id:
                raise ValueError(f"у термина №{index + 1} нет id")
            if term_id in ids:
                raise ValueError(f"повторяющийся id термина: {term_id}")
            ids.add(term_id)

            canonical = normalize(str(raw.get("canonical") or ""))
            if not canonical:
                raise ValueError(f"у термина {term_id} нет canonical")
            kind = str(raw.get("kind") or "term")
            default_replace = bool(raw.get("replace", True))

            raw_aliases = raw.get("aliases")
            if not isinstance(raw_aliases, list) or not raw_aliases:
                raise ValueError(f"у термина {term_id} нет aliases")

            aliases: list[Alias] = []
            seen: set[str] = set()
            for item in raw_aliases:
                replace = default_replace
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = str(item.get("text") or "")
                    if "replace" in item:
                        replace = bool(item["replace"])
                else:
                    raise ValueError(f"у термина {term_id} неверный алиас")
                alias = normalize(text)
                if not alias:
                    raise ValueError(f"у термина {term_id} пустой алиас")
                if alias in seen:
                    continue
                seen.add(alias)
                aliases.append(Alias(alias, replace))
            terms.append(Term(term_id, canonical, kind, tuple(aliases)))
        return cls(terms)

    @classmethod
    def load(cls, path: Path | None = None) -> Dictionary:
        path = Path(path or DICTIONARY_PATH)
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            log.warning("Словарь не найден: %s", path)
            return cls()
        except OSError as exc:
            log.warning("Не удалось прочитать словарь %s: %s", path, exc)
            return cls()
        try:
            return cls.from_data(json.loads(raw))
        except ValueError as exc:
            log.warning("Словарь %s пропущен: %s", path, exc)
            return cls()

    def apply(self, text: str) -> str:
        """Заменяет безопасные варианты названий на канонические."""
        if not text or self._replace_re is None:
            return text
        replacements = self._replacements

        def replace(match: re.Match[str]) -> str:
            key = normalize(match.group(0))
            return replacements.get(key, match.group(0))

        return self._replace_re.sub(replace, text)

    def find(self, text: str) -> list[TermMatch]:
        """Находит все термины в тексте, включая не заменяемые алиасы."""
        if not text or self._find_re is None:
            return []
        matches: list[TermMatch] = []
        for match in self._find_re.finditer(text):
            item = self._by_alias.get(normalize(match.group(0)))
            if item is None:
                continue
            term, alias = item
            matches.append(TermMatch(term, alias, match.start(), match.end()))
        return matches

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _compile(aliases: Iterable[str]) -> re.Pattern[str] | None:
        texts = sorted(set(aliases), key=lambda text: (-len(text), text))
        if not texts:
            return None
        parts = []
        for text in texts:
            escaped = re.escape(text)
            # normalize() схлопывает пробелы, но во входе их может быть больше.
            escaped = escaped.replace(r"\ ", r"\s+").replace(" ", r"\s+")
            parts.append(escaped)
        return re.compile(
            r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.UNICODE
        )


_dictionary: Dictionary | None = None


def get_dictionary() -> Dictionary:
    global _dictionary
    if _dictionary is None:
        _dictionary = Dictionary.load()
    return _dictionary


def reset_dictionary() -> None:
    """Сбрасывает кэш: нужен тестам и ручной перезагрузке словаря."""
    global _dictionary
    _dictionary = None


def apply(text: str) -> str:
    return get_dictionary().apply(text)


def find(text: str) -> list[TermMatch]:
    return get_dictionary().find(text)

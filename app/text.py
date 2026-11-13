from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s.]+", re.UNICODE)
_EDGE_DOT_RE = re.compile(r"(?<!\w)\.|\.(?!\w)", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)

# Словесные числительные. Формы даны в нижнем регистре и без «ё»:
# normalize() уже привёл текст к такому виду.
_NUMBER_SMALL: dict[str, int] = {
    "ноль": 0,
    "нуль": 0,
    "один": 1,
    "одна": 1,
    "одну": 1,
    "одного": 1,
    "одной": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "двум": 2,
    "двумя": 2,
    "три": 3,
    "трех": 3,
    "трем": 3,
    "тремя": 3,
    "четыре": 4,
    "четырех": 4,
    "четырем": 4,
    "четырьмя": 4,
    "пять": 5,
    "пяти": 5,
    "пятью": 5,
    "шесть": 6,
    "шести": 6,
    "шестью": 6,
    "семь": 7,
    "семи": 7,
    "семью": 7,
    "восемь": 8,
    "восьми": 8,
    "восемью": 8,
    "девять": 9,
    "девяти": 9,
    "девятью": 9,
    "десять": 10,
    "десяти": 10,
    "десятью": 10,
    "одиннадцать": 11,
    "одиннадцати": 11,
    "одиннадцатью": 11,
    "двенадцать": 12,
    "двенадцати": 12,
    "двенадцатью": 12,
    "тринадцать": 13,
    "тринадцати": 13,
    "тринадцатью": 13,
    "четырнадцать": 14,
    "четырнадцати": 14,
    "четырнадцатью": 14,
    "пятнадцать": 15,
    "пятнадцати": 15,
    "пятнадцатью": 15,
    "шестнадцать": 16,
    "шестнадцати": 16,
    "шестнадцатью": 16,
    "семнадцать": 17,
    "семнадцати": 17,
    "семнадцатью": 17,
    "восемнадцать": 18,
    "восемнадцати": 18,
    "восемнадцатью": 18,
    "девятнадцать": 19,
    "девятнадцати": 19,
    "девятнадцатью": 19,
}

_NUMBER_TENS: dict[str, int] = {
    "двадцать": 20,
    "двадцати": 20,
    "двадцатью": 20,
    "тридцать": 30,
    "тридцати": 30,
    "тридцатью": 30,
    "сорок": 40,
    "сорока": 40,
    "пятьдесят": 50,
    "пятидесяти": 50,
    "пятьюдесятью": 50,
    "шестьдесят": 60,
    "шестидесяти": 60,
    "шестьюдесятью": 60,
    "семьдесят": 70,
    "семидесяти": 70,
    "семьюдесятью": 70,
    "восемьдесят": 80,
    "восьмидесяти": 80,
    "восемьюдесятью": 80,
    "девяносто": 90,
    "девяноста": 90,
}

_NUMBER_HUNDREDS: dict[str, int] = {
    "сто": 100,
    "ста": 100,
    "двести": 200,
    "двухсот": 200,
    "двумстам": 200,
    "двумястами": 200,
    "триста": 300,
    "трехсот": 300,
    "тремстам": 300,
    "тремястами": 300,
    "четыреста": 400,
    "четырехсот": 400,
    "четыремстам": 400,
    "четырьмястами": 400,
    "пятьсот": 500,
    "пятисот": 500,
    "пятистам": 500,
    "пятьюстами": 500,
    "шестьсот": 600,
    "шестисот": 600,
    "шестистам": 600,
    "шестьюстами": 600,
    "семьсот": 700,
    "семисот": 700,
    "семистам": 700,
    "семьюстами": 700,
    "восемьсот": 800,
    "восьмисот": 800,
    "восьмистам": 800,
    "восемьюстами": 800,
    "девятьсот": 900,
    "девятисот": 900,
    "девятистам": 900,
    "девятьюстами": 900,
}

_NUMBER_SCALES: dict[str, int] = {
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
    "тысячу": 1000,
    "тысячей": 1000,
    "тысяче": 1000,
    "тысячам": 1000,
    "тысячами": 1000,
    "тысячах": 1000,
    "миллион": 10**6,
    "миллиона": 10**6,
    "миллионов": 10**6,
    "миллиону": 10**6,
    "миллионом": 10**6,
    "миллионе": 10**6,
    "миллионы": 10**6,
    "миллионам": 10**6,
    "миллионами": 10**6,
    "миллионах": 10**6,
    "миллиард": 10**9,
    "миллиарда": 10**9,
    "миллиардов": 10**9,
    "миллиарду": 10**9,
    "миллиардом": 10**9,
    "миллиарде": 10**9,
    "миллиарды": 10**9,
    "миллиардам": 10**9,
    "миллиардами": 10**9,
    "миллиардах": 10**9,
}

_NUMBER_SCALE_RANK = {1000: 1, 10**6: 2, 10**9: 3}
_NUMBER_WORDS = (
    frozenset(_NUMBER_SMALL)
    | frozenset(_NUMBER_TENS)
    | frozenset(_NUMBER_HUNDREDS)
    | frozenset(_NUMBER_SCALES)
)

# Порядковые числительные: «двадцать первый» нельзя превращать в «20 первый».
_ORDINAL_WORDS = frozenset(
    {
        "первый",
        "первая",
        "первое",
        "первые",
        "первую",
        "первой",
        "первого",
        "первому",
        "первым",
        "первом",
        "первых",
        "второй",
        "вторая",
        "второе",
        "вторые",
        "вторую",
        "второго",
        "второму",
        "вторым",
        "втором",
        "вторых",
        "третий",
        "третья",
        "третье",
        "третьи",
        "третью",
        "третьей",
        "третьего",
        "третьему",
        "третьим",
        "третьем",
        "третьих",
        "четвертый",
        "четвертая",
        "четвертое",
        "четвертые",
        "четвертую",
        "четвертого",
        "четвертому",
        "четвертым",
        "четвертом",
        "четвертых",
        "пятого",
        "пятому",
        "пятым",
        "пятом",
        "пятых",
        "пятый",
        "пятая",
        "пятое",
        "пятые",
        "пятую",
        "шестой",
        "шестая",
        "шестое",
        "шестые",
        "шестую",
        "шестого",
        "шестому",
        "шестым",
        "шестом",
        "шестых",
        "седьмой",
        "седьмая",
        "седьмое",
        "седьмые",
        "седьмую",
        "седьмого",
        "седьмому",
        "седьмым",
        "седьмом",
        "седьмых",
        "восьмой",
        "восьмая",
        "восьмое",
        "восьмые",
        "восьмую",
        "восьмого",
        "восьмому",
        "восьмым",
        "восьмом",
        "восьмых",
        "девятый",
        "девятая",
        "девятое",
        "девятые",
        "девятую",
        "девятого",
        "девятому",
        "девятым",
        "девятом",
        "девятых",
        "десятый",
        "десятая",
        "десятое",
        "десятые",
        "десятую",
        "десятого",
        "десятому",
        "десятым",
        "десятом",
        "десятых",
        "двадцатый",
        "тридцатый",
        "сороковой",
        "пятидесятый",
        "шестидесятый",
        "семидесятый",
        "восьмидесятый",
        "девяностый",
        "сотый",
        "тысячный",
        "миллионный",
    }
)


def _parse_number_words(words: list[str]) -> int | None:
    """Собирает числительное из слов: «восемьсот двенадцать» → 812.

    Слова должны идти в правильном порядке (сотни → десятки → единицы),
    иначе «один два» не превратится в 3.
    """
    total = 0
    current = 0
    group_seen = False
    group_rank = 99
    scale_rank = 99
    seen = False
    for word in words:
        scale = _NUMBER_SCALES.get(word)
        if scale is not None:
            rank = _NUMBER_SCALE_RANK[scale]
            if rank >= scale_rank:
                return None
            total += (current if group_seen else 1) * scale
            current = 0
            group_seen = False
            group_rank = 99
            scale_rank = rank
            seen = True
            continue

        value = _NUMBER_HUNDREDS.get(word)
        if value is not None:
            rank = 3
        else:
            value = _NUMBER_TENS.get(word)
            if value is not None:
                rank = 2
            else:
                value = _NUMBER_SMALL.get(word)
                if value is None:
                    return None
                rank = 2 if value >= 10 else 1
        if rank >= group_rank:
            return None
        current += value
        group_seen = True
        group_rank = rank
        seen = True
    if not seen:
        return None
    return total + current


def numbers_to_digits(text: str) -> str:
    """Заменяет словесные числа на цифры: «восемьсот двенадцать» → «812».

    Текст должен быть уже нормализован: нижний регистр, без «ё» и пунктуации.
    Порядковые числительные («двадцать первый») не трогаем.
    """
    if not text:
        return text
    tokens = text.split()
    result: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] not in _NUMBER_WORDS:
            result.append(tokens[index])
            index += 1
            continue
        end = index
        while end < len(tokens) and tokens[end] in _NUMBER_WORDS:
            end += 1
        if end < len(tokens) and tokens[end] in _ORDINAL_WORDS:
            result.extend(tokens[index:end])
        else:
            value = _parse_number_words(tokens[index:end])
            if value is None:
                result.extend(tokens[index:end])
            else:
                result.append(str(value))
        index = end
    return " ".join(result)


def normalize(text: str) -> str:
    """Приводит распознанный текст к виду, удобному для сравнения.

    Точка сохраняется внутри слов (example.com), но убирается на границах.
    Словесные числа заменяются цифрами: «восемьсот двенадцать» → «812».
    """
    text = (text or "").lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    text = _EDGE_DOT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return numbers_to_digits(text)


_TRANSLIT_TABLE = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def translit(text: str) -> str:
    """Переводит кириллицу в латиницу для сравнения названий треков.

    Названия на SoundCloud часто английские, а STT слышит их по-русски:
    «ева ева» должно совпасть с «EVA EVA».
    """
    return (text or "").lower().replace("ё", "е").translate(_TRANSLIT_TABLE)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def fuzzy_contains(text: str, phrase: str, max_distance: int = 2) -> bool:
    """Ищет фразу в тексте, допуская небольшие ошибки распознавания."""
    if not text or not phrase:
        return False
    if phrase in text:
        return True
    words = text.split()
    target = phrase.split()
    size = len(target)
    if size == 0 or len(words) < size:
        return False
    for start in range(len(words) - size + 1):
        window = " ".join(words[start : start + size])
        if levenshtein(window, phrase) <= max_distance:
            return True
    return False


_SOUNDCLOUD_ANY_RE = re.compile(
    r"(?:сау[нтдк]?[ао]?|с[ао]ун[дт]?|с[ао]н[дт]?[ао]?)"
    r"\s*[кг]?\s*[лр][ао]у?н?[дт]\w*"
    r"|sound\s*cloud",
    re.UNICODE,
)
_LOFI_ANY_RE = re.compile(r"(?:лоу|лу|ло)\s*фай\w*", re.UNICODE)
_YOUTUBE_SHORT_RE = re.compile(r"(открой|включи|запусти|поставь)\s+(?:ту|тю|тюб|ют|юта|юту|юбе|юб)")
_SOUNDCLOUD_ENDINGS = "ауе"
_QUIET_NUMBER_RE = re.compile(
    r"^тишина\s+(?=\d|ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять|десять)",
    re.UNICODE,
)
# «случайно лайкнутый», «рандомную лайкнутую песню» — GigaAM часто путает
# прилагательное с наречием. Приводим к одной форме, которую ждут интенты.
_LIKES_RANDOM_RE = re.compile(
    r"\b(?:случайн\w*|рандомн\w*)\s+"
    r"(?:лайкнут\w*|любим\w*|избранн\w*)"
    r"(?:\s+(?:трек|песн\w*|музык\w*))?",
    re.UNICODE,
)


def _canonical_soundcloud(match: re.Match[str]) -> str:
    word = match.group(0).replace(" ", "")
    ending = word[-1] if word[-1] in _SOUNDCLOUD_ENDINGS else ""
    return "саундклауд" + ending


def canonicalize(text: str) -> str:
    """Приводит названия сервисов к одному написанию.

    STT слышит бренд по-разному: «санклауда», «саунглауда», «соунклауда».
    Без нормализации страдает и точность текста, и поисковый запрос.
    """
    if not text:
        return text
    result = _SOUNDCLOUD_ANY_RE.sub(_canonical_soundcloud, text)
    result = _LIKES_RANDOM_RE.sub("случайный лайкнутый трек", result)
    result = _LOFI_ANY_RE.sub("лоуфай", result)
    if result.startswith("тишина "):
        # GigaAM слышит «тише» как «тишина»: «тишина два» → «тише на два».
        result = re.sub(r"^тишина\s+на\s+", "тише на ", result)
        result = _QUIET_NUMBER_RE.sub("тише на ", result)
    match = _YOUTUBE_SHORT_RE.fullmatch(result)
    if match:
        result = f"{match.group(1)} ютуб"
    # Словарь предметной области дополняет точечные правила выше: названия
    # сервисов и их искажения. Импорт локальный, чтобы не было цикла:
    # app.dictionary использует normalize() из этого модуля.
    from .dictionary import apply as apply_dictionary

    return apply_dictionary(result)


_WAKE_FILLERS = {
    "окей",
    "ок",
    "эй",
    "ассистент",
    "ассистэнт",
    "хей",
    "хэй",
    "хай",
}

# Слова, которые STT иногда слышит перед ключевой фразой: pre-roll
# захватывает хвост музыки или вдох. «пять хэллоуина стоп» → «стоп».
_WAKE_NOISE_WORDS = {
    "а",
    "и",
    "ну",
    "вот",
    "ох",
    "ай",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
}

# Начала, по которым STT узнаёт «хей луна» / «эй луна». Нужны только для
# широкого допуска в начале фразы: обычные команды («включи», «открой»)
# под него не попадают.
_WAKE_ONSETS = ("хей", "хэй", "хай", "хэ", "хе", "ха", "эй", "ей")


def _wake_distance_limit(window: str, target: str, base: int) -> int:
    """Насколько сильно услышанное может отличаться от алиаса.

    GigaAM слышит «хей луна» как «хэллоуин», «хэллона» или «хаймена».
    Для похожего начала допускаем больше ошибок, но ограничиваем длину,
    чтобы не принять за wake-фразу обычную команду.
    """
    if not window.startswith(_WAKE_ONSETS):
        return base
    if not 5 <= len(window) <= len(target) + 2:
        return base
    return max(base, 5)


def _wake_prefix_size(text: str, aliases: list[str], max_distance: int = 2) -> int:
    """Сколько первых слов занимает ключевая фраза.

    STT часто сливает «хей луна» в «хейлуна», «эйлона» или даже «эйна»,
    поэтому сравниваем первые 1–3 слова без пробелов с тем же вариантом
    алиаса. Для похожего начала («хэллоуин», «хаймена») допуск шире:
    иначе искажённая wake-фраза остаётся в тексте команды.
    """
    words = text.split()
    targets = [normalize(alias).replace(" ", "") for alias in aliases]
    targets = [target for target in targets if target]
    if not targets:
        return 0
    best_size = 0
    best_distance: int | None = None
    for size in (1, 2, 3):
        if len(words) < size:
            continue
        window = "".join(words[:size])
        for target in targets:
            limit = _wake_distance_limit(window, target, max_distance)
            distance = levenshtein(window, target)
            if abs(len(window) - len(target)) > limit or distance > limit:
                continue
            # «хайлу на …»: слово по отдельности похоже на алиас, но вместе
            # со следующим словом совпадает гораздо точнее. Берём вариант
            # с наименьшим расстоянием, а не первый подходящий.
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_size = size
    return best_size


def _wake_noise_prefix_size(text: str, aliases: list[str]) -> int:
    """Ищет wake-фразу после слова-шума в начале.

    Pre-roll иногда захватывает «пять» из музыки: «пять хэллоуина стоп».
    Тогда команда начинается не с wake-фразы, и обычный префиксный разбор
    её не видит.
    """
    words = text.split()
    if len(words) < 2 or words[0] not in _WAKE_NOISE_WORDS:
        return 0
    size = _wake_prefix_size(" ".join(words[1:]), aliases)
    return 1 + size if size else 0


def strip_wake_word(text: str, aliases: list[str]) -> str:
    """Убирает ключевую фразу в начале, допуская ошибки распознавания.

    STT иногда теряет часть фразы («окей открой ютуб») или сливает её в одно
    слово («хейлуна открой ютуб»), поэтому после точного совпадения
    проверяются слитные варианты, шум перед фразой и отдельные слова-wake.
    """
    normalized = normalize(text)
    prepared = sorted(
        (normalize(alias) for alias in aliases if normalize(alias)),
        key=len,
        reverse=True,
    )
    for alias in prepared:
        if normalized.startswith(alias):
            return normalized[len(alias) :].strip()

    words = normalized.split()
    size = _wake_prefix_size(normalized, prepared)
    if size:
        return " ".join(words[size:]).strip()

    size = _wake_noise_prefix_size(normalized, prepared)
    if size:
        return " ".join(words[size:]).strip()

    while words and words[0] in _WAKE_FILLERS:
        words.pop(0)
    remainder = " ".join(words)
    size = _wake_prefix_size(remainder, prepared)
    if size:
        return " ".join(words[size:]).strip()
    size = _wake_noise_prefix_size(remainder, prepared)
    if size:
        return " ".join(words[size:]).strip()
    return remainder


def split_agent_trigger(text: str, triggers: list[str]) -> tuple[bool, str]:
    """Отделяет явный вызов агента от текста запроса.

    Триггер учитывается только в начале фразы, поэтому «включи фильм
    агент 007» остаётся обычной командой. Возвращает пару
    «сработал ли триггер» и остаток запроса.
    """
    normalized = normalize(text)
    prepared = {normalize(trigger) for trigger in triggers}
    prepared.discard("")
    for trigger in sorted(prepared, key=len, reverse=True):
        if normalized == trigger:
            return True, ""
        if normalized.startswith(trigger + " "):
            return True, normalized[len(trigger) :].strip()
    return False, text

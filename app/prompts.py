"""Промпты ответов облачной модели.

Тексты общие для любых провайдеров: клиент только подставляет их в
system/user-сообщения. Ответы рассчитаны на озвучку, поэтому без markdown
и списков.
"""

from __future__ import annotations


def _answer_prompt(length: str) -> str:
    return f"""Ты — голосовой ассистент. Ответь на вопрос {length}, простым русским языком.
Правила:
1. Если не знаешь ответа или вопрос требует свежих данных (новости, курс, цены, погода), скажи: «Не знаю, стоит поискать в интернете».
2. Не выдумывай числа, даты, имена и события.
3. Пиши только по-русски. Не используй иероглифы и другие алфавиты, кроме имён и названий.
4. Без markdown, списков и пояснений вокруг ответа.
"""


ANSWER_SYSTEM = _answer_prompt("кратко: 2–4 предложения")
ANSWER_SYSTEM_SHORT = _answer_prompt("совсем кратко: 1–2 предложения")
ANSWER_SYSTEM_DETAILED = _answer_prompt("подробно: 4–6 предложений")


def _rag_prompt(length: str) -> str:
    return (
        f"Ответь на вопрос по источникам ниже: {length}.\n"
        "Ставь ссылку [1] или [2] после каждого факта.\n"
        "Пиши сразу факт, без фраз «в источнике», «на сайте», «было сообщение».\n"
        "Не выдумывай. Если ответа нет, скажи: «Не знаю».\n"
    )


def _news_prompt(count: str, sentences: str) -> str:
    return f"""Выпиши из источников ниже {count} главных новостей.
Каждую новость пиши с новой строки, {sentences}, в начале номер источника: [n].
Выбери по одному главному факту из разных источников.
Не добавляй пояснений и оценок. Не выдумывай.
"""


RAG_SYSTEM = _rag_prompt("кратко, 2–4 предложения")
RAG_SYSTEM_SHORT = _rag_prompt("совсем кратко, 1–2 предложения")
RAG_SYSTEM_NEWS_SHORT = _news_prompt("5–7", "1–2 предложения")
RAG_SYSTEM_NEWS = _news_prompt("3–4", "3–4 предложения")
RAG_SYSTEM_NEWS_DETAILED = _news_prompt("5–7", "2–3 предложения")
RAG_SYSTEM_ARTICLE = """Составь подробный пересказ новости по источникам ниже.
Используй важные детали: цифры, сроки, участников и планы.
Напиши 4–6 предложений, каждое со ссылкой [1] или [2].
Если источники расходятся в числах, укажи оба значения со ссылками.
Пиши сразу факты, без вступлений и оценок. Не выдумывай.
"""
RAG_SYSTEM_DETAILED = _rag_prompt("подробно, 4–6 предложений")

RAG_PROMPTS = {
    "short": RAG_SYSTEM_SHORT,
    "news_short": RAG_SYSTEM_NEWS_SHORT,
    "news": RAG_SYSTEM_NEWS,
    "news_detailed": RAG_SYSTEM_NEWS_DETAILED,
    "article": RAG_SYSTEM_ARTICLE,
    "detailed": RAG_SYSTEM_DETAILED,
}

ANSWER_PROMPTS = {
    "short": ANSWER_SYSTEM_SHORT,
    "detailed": ANSWER_SYSTEM_DETAILED,
}

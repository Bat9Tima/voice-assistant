from __future__ import annotations

import unittest

from app.stt import (
    EnglishSpan,
    Transcript,
    Word,
    _gap_spans,
    _gigaam_words,
    _merge_with_spans,
    merge_transcripts,
)


def w(text: str, start: float, end: float, probability: float = 1.0) -> Word:
    return Word(text=text, start=start, end=end, probability=probability)


def words(*items: tuple[str, float, float]) -> list[Word]:
    return [w(text, start, end) for text, start, end in items]


class GigaAMWordsTest(unittest.TestCase):
    def test_words_are_built_from_character_tokens(self) -> None:
        tokens = ["н", "а", "й", "д", "и", " ", "ф", "о", "н", "и"]
        timestamps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.2, 2.5, 2.6, 2.7]
        result = _gigaam_words(tokens, timestamps)
        self.assertEqual([word.text for word in result], ["найди", "фони"])
        self.assertAlmostEqual(result[0].start, 0.2)
        self.assertAlmostEqual(result[0].end, 0.7)
        self.assertAlmostEqual(result[1].start, 2.2)
        self.assertAlmostEqual(result[1].end, 2.8)


class MergeTranscriptsTest(unittest.TestCase):
    def test_pure_russian_keeps_russian_text(self) -> None:
        ru = Transcript(
            text="найди смешных котов",
            words=words(
                ("найди", 0.2, 0.7),
                ("смешных", 0.8, 1.5),
                ("котов", 1.6, 2.1),
            ),
            language="ru",
        )
        en = Transcript(
            text="найди смешных котов",
            words=words(
                ("найди", 0.2, 0.7),
                ("смешных", 0.8, 1.5),
                ("котов", 1.6, 2.1),
            ),
            language="ru",
        )
        self.assertEqual(merge_transcripts(ru, en), "найди смешных котов")

    def test_pure_english_uses_english_text(self) -> None:
        ru = Transcript(
            text="файв фони кэтс",
            words=words(("файв", 0.1, 0.4), ("фони", 0.5, 0.9), ("кэтс", 1.0, 1.4)),
            language="ru",
            confidence=0.70,
        )
        en = Transcript(
            text="Find Funny Cats",
            words=[
                w("Find", 0.1, 0.4, 0.6),
                w("Funny", 0.5, 0.9, 0.5),
                w("Cats", 1.0, 1.4, 0.6),
            ],
            language="en",
        )
        self.assertEqual(merge_transcripts(ru, en), "Find Funny Cats")

    def test_mixed_phrase_replaces_english_part(self) -> None:
        ru = Transcript(
            text="найди фони кэтс",
            words=words(
                ("найди", 0.2, 0.7),
                ("фони", 2.2, 2.8),
                ("кэтс", 2.9, 3.5),
            ),
            language="ru",
        )
        en = Transcript(
            text="Find Funny Cats",
            words=[
                w("Find", 0.1, 0.7, 0.5),
                w("Funny", 2.7, 3.0, 0.9),
                w("Cats", 3.0, 3.6, 0.8),
            ],
            language="en",
        )
        self.assertEqual(merge_transcripts(ru, en), "найди Funny Cats")

    def test_mixed_phrase_keeps_youtube_suffix(self) -> None:
        ru = Transcript(
            text="найди на ютубе фони кэтс",
            words=words(
                ("найди", 0.2, 0.7),
                ("на", 0.7, 0.9),
                ("ютубе", 1.0, 1.5),
                ("фони", 2.2, 2.8),
                ("кэтс", 2.9, 3.5),
            ),
            language="ru",
        )
        en = Transcript(
            text="Find Funny Cats on YouTube",
            words=[
                w("Find", 0.1, 0.7, 0.5),
                w("Funny", 2.7, 3.0, 0.9),
                w("Cats", 3.0, 3.6, 0.8),
                w("on", 3.7, 3.9, 0.9),
                w("YouTube", 3.9, 4.4, 0.9),
            ],
            language="en",
        )
        self.assertEqual(merge_transcripts(ru, en), "найди на ютубе Funny Cats")

    def test_mixed_phrase_keeps_soundcloud_suffix(self) -> None:
        ru = Transcript(
            text="включи богемская рапсодия на саундклауде",
            words=words(
                ("включи", 0.2, 0.8),
                ("богемская", 1.0, 1.8),
                ("рапсодия", 1.9, 2.7),
                ("на", 2.9, 3.0),
                ("саундклауде", 3.0, 3.7),
            ),
            language="ru",
        )
        en = Transcript(
            text="Play Bohemian Rhapsody",
            words=[
                w("Play", 0.1, 0.8, 0.6),
                w("Bohemian", 1.0, 1.7, 0.8),
                w("Rhapsody", 1.8, 2.6, 0.9),
            ],
            language="en",
        )
        self.assertEqual(
            merge_transcripts(ru, en),
            "включи Bohemian Rhapsody на саундклауде",
        )

    def test_brand_at_end_is_replaced(self) -> None:
        ru = Transcript(
            text="найди кроссовки найк",
            words=words(
                ("найди", 0.2, 0.7),
                ("кроссовки", 1.0, 1.8),
                ("найк", 1.9, 2.3),
            ),
            language="ru",
        )
        en = Transcript(
            text="Find Nike",
            words=[w("Find", 0.1, 0.7, 0.6), w("Nike", 1.9, 2.3, 0.9)],
            language="en",
        )
        self.assertEqual(merge_transcripts(ru, en), "найди Nike")

    def test_english_not_heard_by_gigaam_is_appended_before_site(self) -> None:
        ru = Transcript(
            text="включи на саундклауде",
            words=words(
                ("включи", 0.2, 0.8),
                ("на", 1.0, 1.1),
                ("саундклауде", 1.2, 1.9),
            ),
            language="ru",
        )
        en = Transcript(
            text="Play Bohemian Rhapsody",
            words=[
                w("Play", 1.0, 1.4, 0.6),
                w("Bohemian", 1.5, 2.2, 0.8),
                w("Rhapsody", 2.2, 2.9, 0.9),
            ],
            language="en",
        )
        self.assertEqual(
            merge_transcripts(ru, en),
            "включи Bohemian Rhapsody на саундклауде",
        )

    def test_empty_english_keeps_russian(self) -> None:
        ru = Transcript(text="открой ютуб", words=words(("открой", 0.2, 0.7)))
        self.assertEqual(merge_transcripts(ru, None), "открой ютуб")

    def test_confident_russian_ignores_english_hallucination(self) -> None:
        cases = (
            (
                "найди смешных котов",
                words(("найди", 0.24, 0.68), ("смешных", 0.84, 1.44), ("котов", 1.6, 2.18)),
                "in the next few months.",
            ),
            (
                "открой ютуб",
                words(("открой", 0.2, 0.76), ("ютуб", 0.92, 1.42)),
                "at growing YouTube.",
            ),
            (
                "найди",
                words(("найди", 0.24, 0.78)),
                "Najee.",
            ),
        )
        for text, ru_words, en_text in cases:
            with self.subTest(text=text):
                ru = Transcript(text=text, words=ru_words, language="ru", confidence=0.92)
                en = Transcript(text=en_text, language="en", avg_logprob=-0.9)
                self.assertEqual(merge_transcripts(ru, en), text)

    def test_confident_russian_keeps_only_gap_span(self) -> None:
        ru = Transcript(
            text="найди на ютубе",
            words=words(
                ("найди", 0.24, 0.68),
                ("на", 0.8, 0.96),
                ("ютубе", 1.16, 1.82),
            ),
            language="ru",
            confidence=0.887,
        )
        spans = [
            EnglishSpan(text="in the you to", start=0.7, end=1.95),
            EnglishSpan(text="Funny cats", start=3.36, end=4.5),
        ]
        accepted = _gap_spans(ru, spans)
        self.assertEqual([span.text for span in accepted], ["Funny cats"])

    def test_russian_overlap_is_allowed_when_not_confident(self) -> None:
        ru = Transcript(
            text="найди смешных котов",
            words=words(("найди", 0.24, 0.68), ("смешных", 0.84, 1.44)),
            language="ru",
            confidence=0.70,
        )
        spans = [EnglishSpan(text="funny", start=0.8, end=1.4)]
        self.assertEqual(len(_gap_spans(ru, spans)), 1)

    def test_short_english_hallucination_is_not_appended(self) -> None:
        ru = Transcript(
            text="найди",
            words=words(("найди", 0.12, 0.62)),
            language="ru",
            confidence=0.80,
        )
        en = Transcript(text="Bye-bye.", language="en", avg_logprob=-0.4)
        self.assertEqual(merge_transcripts(ru, en), "найди")

    def test_uncertain_english_is_not_appended(self) -> None:
        ru = Transcript(
            text="найди смешных котов",
            words=words(("найди", 0.12, 0.52), ("смешных", 0.6, 1.08), ("котов", 1.16, 1.62)),
            language="ru",
            confidence=0.80,
        )
        en = Transcript(text="in the next few months", language="en", avg_logprob=-1.2)
        self.assertEqual(merge_transcripts(ru, en), "найди смешных котов")

    def test_two_word_english_query_is_appended(self) -> None:
        ru = Transcript(
            text="найди",
            words=words(("найди", 0.12, 0.62)),
            language="ru",
            confidence=0.80,
        )
        en = Transcript(text="Funny cats.", language="en", avg_logprob=-0.3)
        self.assertEqual(merge_transcripts(ru, en), "найди Funny cats")

    def test_site_word_merged_with_english_keeps_site(self) -> None:
        ru = Transcript(
            text="найди на ютубекадс",
            words=words(
                ("найди", 0.12, 0.48),
                ("на", 0.6, 0.68),
                ("ютубекадс", 0.84, 3.34),
            ),
            language="ru",
            confidence=0.796,
        )
        spans = [EnglishSpan(text="Funny cats", start=2.49, end=3.36)]
        self.assertEqual(
            _merge_with_spans(ru, spans),
            "найди на ютубе Funny cats",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.embeddings import EmbeddingError, VoyageEmbedder


class FakeTransport:
    """Подмена HTTP: запоминает payload и отдаёт заготовленный ответ."""

    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {}
        self.error = error
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


def response(*vectors: list[float]) -> dict:
    return {
        "model": "voyage-4",
        "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
        "usage": {"total_tokens": 10},
    }


class VoyageEmbedderTest(unittest.TestCase):
    def make(self, transport: FakeTransport) -> VoyageEmbedder:
        return VoyageEmbedder("mongodb/voyage-4", api_key="key", transport=transport)

    def test_document_payload_and_vectors(self) -> None:
        transport = FakeTransport(response([0.1, 0.2], [0.3, 0.4]))
        embedder = self.make(transport)

        vectors = embedder.embed(["раз", "два"], input_type="document")

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        payload = transport.payloads[0]
        self.assertEqual(payload["model"], "mongodb/voyage-4")
        self.assertEqual(payload["input"], ["раз", "два"])
        self.assertEqual(payload["input_type"], "document")

    def test_query_input_type(self) -> None:
        transport = FakeTransport(response([1.0]))
        self.make(transport).embed(["вопрос"], input_type="query")
        self.assertEqual(transport.payloads[0]["input_type"], "query")

    def test_vectors_follow_indexes(self) -> None:
        transport = FakeTransport(
            {
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            }
        )
        self.assertEqual(
            self.make(transport).embed(["а", "б"]),
            [[1.0], [2.0]],
        )

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            self.make(FakeTransport()).embed(["  "])
        with self.assertRaises(EmbeddingError):
            self.make(FakeTransport()).embed([])

    def test_missing_key_is_rejected(self) -> None:
        embedder = VoyageEmbedder("model", api_key="", transport=FakeTransport())
        self.assertFalse(embedder.available())
        with self.assertRaises(EmbeddingError):
            embedder.embed(["привет"])

    def test_transport_error_becomes_embedding_error(self) -> None:
        transport = FakeTransport(error=OSError("нет сети"))
        with self.assertRaises(EmbeddingError):
            self.make(transport).embed(["привет"])

    def test_wrong_number_of_vectors(self) -> None:
        transport = FakeTransport(response([1.0]))
        with self.assertRaises(EmbeddingError):
            self.make(transport).embed(["раз", "два"])

    def test_empty_vector_is_rejected(self) -> None:
        transport = FakeTransport(response([]))
        with self.assertRaises(EmbeddingError):
            self.make(transport).embed(["привет"])

    def test_from_config_uses_cloud_key_and_model(self) -> None:
        embedder = VoyageEmbedder.from_config(
            {
                "memory": {"model": "custom/model", "timeout_sec": 3},
                "cloud_llm": {
                    "base_url": "https://example.test/v1",
                    "api_key": "cloud-key",
                },
            }
        )
        self.assertEqual(embedder.model, "custom/model")
        self.assertEqual(embedder.base_url, "https://example.test/v1")
        self.assertEqual(embedder.api_key, "cloud-key")
        self.assertEqual(embedder.timeout, 3.0)

    def test_own_key_wins_over_cloud(self) -> None:
        embedder = VoyageEmbedder.from_config(
            {
                "memory": {"api_key": "memory-key"},
                "cloud_llm": {"api_key": "cloud-key"},
                "tts": {"api_key": "tts-key"},
            }
        )
        self.assertEqual(embedder.api_key, "memory-key")

    def test_falls_back_to_tts_key(self) -> None:
        embedder = VoyageEmbedder.from_config({"tts": {"api_key": "tts-key"}})
        self.assertEqual(embedder.api_key, "tts-key")


if __name__ == "__main__":
    unittest.main()

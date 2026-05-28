from llm_kg.embeddings.mock_client import MockEmbeddingClient


def test_mock_embeddings_are_deterministic_and_dimensioned() -> None:
    client = MockEmbeddingClient(dimensions=1536)

    first = client.embed_text("SB 330 affects Housing Project Alpha")
    second = client.embed_text("SB 330 affects Housing Project Alpha")

    assert first == second
    assert len(first) == 1536
    assert any(value != 0 for value in first)

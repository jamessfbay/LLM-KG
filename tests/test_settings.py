from pathlib import Path

from llm_kg.config import Settings


def test_settings_loads_workspace_toml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_KG_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_KG_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LLM_KG_TOP_K", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "llm_kg.toml").write_text(
        '\n'.join(
            [
                "[llm]",
                'provider = "mock"',
                'openai_model = "gpt-4.1-mini"',
                "",
                "[database]",
                'url = "postgresql://user:pass@localhost:5432/db"',
                "",
                "[embedding]",
                'provider = "mock"',
                'model = "text-embedding-3-small"',
                "dimensions = 1536",
                "",
                "[query]",
                "top_k = 7",
                'default_mode = "basic"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(tmp_path)

    assert settings.workspace == tmp_path
    assert settings.llm_provider == "mock"
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.database_url == "postgresql://user:pass@localhost:5432/db"
    assert settings.embedding_provider == "mock"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.top_k == 7
    assert settings.query_default_mode == "basic"


def test_environment_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "llm_kg.toml").write_text(
        '\n'.join(["[llm]", 'provider = "mock"', "", "[query]", "top_k = 3", ""]),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_KG_PROVIDER", "openai")
    monkeypatch.setenv("LLM_KG_TOP_K", "11")

    settings = Settings.from_env(tmp_path)

    assert settings.llm_provider == "openai"
    assert settings.top_k == 11

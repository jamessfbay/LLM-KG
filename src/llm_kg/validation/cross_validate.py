from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from llm_kg.config import Settings
from llm_kg.models import (
    Claim,
    CrossValidationClaimResult,
    CrossValidationConsensus,
    CrossValidationProviderResult,
    CrossValidationRun,
    Evidence,
)
from llm_kg.models.core import utc_now
from llm_kg.readers.common import stable_id
from llm_kg.storage import JsonlStore


class ClaimValidator(Protocol):
    provider: str
    model: str

    def validate(self, items: list[dict[str, Any]]) -> CrossValidationProviderResult:
        ...


def cross_validate_claims(
    workspace: Path,
    providers: list[str] | None = None,
    limit: int | None = None,
    output_path: Path | None = None,
    validators: dict[str, ClaimValidator] | None = None,
    settings: Settings | None = None,
) -> CrossValidationRun:
    workspace = workspace.resolve()
    settings = settings or Settings.from_env(workspace)
    provider_names = providers or _provider_names(settings.validation_providers)
    jsonl = JsonlStore(workspace)
    claims = jsonl.load("claims.jsonl", Claim)
    evidence = {item.id: item for item in jsonl.load("evidence.jsonl", Evidence)}
    if limit is not None:
        claims = claims[: max(limit, 0)]
    items = [_claim_payload(claim, evidence) for claim in claims]

    validator_map = validators or _build_validators(settings)
    results: list[CrossValidationProviderResult] = []
    for provider in provider_names:
        validator = validator_map.get(provider)
        if validator is None:
            results.append(
                CrossValidationProviderResult(
                    provider=provider,
                    status="error",
                    error=f"Unsupported or unconfigured validator provider: {provider}",
                )
            )
            continue
        try:
            results.append(validator.validate(items))
        except Exception as exc:
            results.append(
                CrossValidationProviderResult(
                    provider=provider,
                    model=getattr(validator, "model", None),
                    status="error",
                    error=str(exc),
                )
            )

    run = CrossValidationRun(
        id=stable_id("cv", f"{workspace}:{utc_now().isoformat()}:{','.join(provider_names)}"),
        providers=provider_names,
        claim_count=len(items),
        results=results,
        consensus=_build_consensus(claims, results),
    )
    jsonl.upsert("cross_validation_runs.jsonl", [run])
    if output_path:
        target = output_path if output_path.is_absolute() else workspace / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run


def _provider_names(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _build_validators(settings: Settings) -> dict[str, ClaimValidator]:
    return {
        "gemini": GeminiClaimValidator(settings.gemini_model),
        "xai": XAIClaimValidator(settings.xai_model),
    }


def _claim_payload(claim: Claim, evidence_by_id: dict[str, Evidence]) -> dict[str, Any]:
    return {
        "claim_id": claim.id,
        "claim": claim.text,
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
        "evidence": [
            {
                "evidence_id": evidence.id,
                "quote": evidence.quote,
                "page_number": evidence.page_number,
                "source_mode": evidence.source_mode,
                "confidence": evidence.confidence,
            }
            for evidence_id in claim.evidence_ids
            if (evidence := evidence_by_id.get(evidence_id)) is not None
        ],
    }


def _build_prompt(items: list[dict[str, Any]]) -> str:
    schema = """
Return only a JSON object with this shape:
{
  "items": [
    {
      "claim_id": "string",
      "supported": true,
      "support_level": "full|partial|unsupported|unclear",
      "citation_ok": true,
      "planning_fact_ok": true,
      "issues": ["short issue text"],
      "rationale": "one sentence"
    }
  ],
  "summary": {
    "supported_count": 0,
    "partial_count": 0,
    "unsupported_count": 0,
    "citation_problem_count": 0,
    "overall_assessment": "short text"
  }
}
""".strip()
    return "\n".join(
        [
            "You are cross-validating source-linked planning knowledge graph extraction.",
            "Judge only whether each claim is directly supported by its quoted evidence.",
            "Do not use outside knowledge.",
            "A claim is fully supported only if the quote directly states the factual content.",
            "Mark citation_ok false when evidence is missing, page_number is missing, or source provenance is unusable.",
            "planning_fact_ok means the claim is useful for planning review, not a generic drawing label.",
            "",
            schema,
            "",
            "Items to validate:",
            json.dumps(items, indent=2),
        ]
    )


class GeminiClaimValidator:
    provider = "gemini"

    def __init__(self, model: str) -> None:
        self.model = model

    def validate(self, items: list[dict[str, Any]]) -> CrossValidationProviderResult:
        key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            return CrossValidationProviderResult(provider=self.provider, model=self.model, status="error", error="GOOGLE_GEMINI_API_KEY is not set.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": _build_prompt(items)}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0},
        }
        data = _post_json(url, {"x-goog-api-key": key, "Content-Type": "application/json"}, payload)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _provider_result(self.provider, self.model, text)


class XAIClaimValidator:
    provider = "xai"

    def __init__(self, model: str) -> None:
        self.model = model

    def validate(self, items: list[dict[str, Any]]) -> CrossValidationProviderResult:
        key = os.getenv("XAI_API_KEY")
        if not key:
            return CrossValidationProviderResult(provider=self.provider, model=self.model, status="error", error="XAI_API_KEY is not set.")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _build_prompt(items)}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data = _post_json(
            "https://api.x.ai/v1/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload,
        )
        text = data["choices"][0]["message"]["content"]
        return _provider_result(self.provider, self.model, text)


def _provider_result(provider: str, model: str, text: str) -> CrossValidationProviderResult:
    parsed = _extract_json(text)
    return CrossValidationProviderResult(
        provider=provider,
        model=model,
        status="ok",
        items=[CrossValidationClaimResult.model_validate(item) for item in parsed.get("items", [])],
        summary=parsed.get("summary", {}),
    )


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _build_consensus(
    claims: list[Claim],
    results: list[CrossValidationProviderResult],
) -> list[CrossValidationConsensus]:
    by_provider = {
        result.provider: {item.claim_id: item for item in result.items}
        for result in results
        if result.status == "ok"
    }
    error_providers = [result.provider for result in results if result.status == "error"]
    consensus: list[CrossValidationConsensus] = []
    for claim in claims:
        validators: list[str] = []
        supported_by: list[str] = []
        flagged_by: list[str] = []
        issues: list[str] = []
        for provider, items in by_provider.items():
            item = items.get(claim.id)
            if item is None:
                continue
            validators.append(provider)
            if item.supported and item.support_level == "full" and item.citation_ok and item.planning_fact_ok:
                supported_by.append(provider)
            else:
                flagged_by.append(provider)
                issues.extend(f"{provider}: {issue}" for issue in item.issues)
                if not item.issues:
                    issues.append(f"{provider}: {item.support_level}")
        if not validators and error_providers:
            verdict = "error"
        elif flagged_by:
            verdict = "needs_review" if supported_by else "rejected"
        elif validators and len(supported_by) == len(validators):
            verdict = "accepted"
        else:
            verdict = "needs_review"
        consensus.append(
            CrossValidationConsensus(
                claim_id=claim.id,
                verdict=verdict,
                validators=validators,
                supported_by=supported_by,
                flagged_by=flagged_by,
                issues=issues,
            )
        )
    return consensus

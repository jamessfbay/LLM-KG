# Extract Claims

Extract factual claims as strict JSON.

Output shape:

```json
{
  "claims": [
    {
      "id": "stable optional id",
      "text": "specific factual claim",
      "source_ids": ["doc_id"],
      "evidence_ids": ["evidence_id"],
      "subject": "optional subject text",
      "predicate": "optional typed predicate",
      "object": "optional object text",
      "confidence": 0.0,
      "status": "active"
    }
  ]
}
```

Rules:
- Every claim must be supported by evidence.
- Avoid vague claims like "this is important."
- Do not invent source text, page numbers, or URLs.
- Use `uncertain` status when evidence is weak or ambiguous.

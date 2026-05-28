# Extract Entities

Extract named entities and stable concepts as strict JSON.

Output shape:

```json
{
  "entities": [
    {
      "id": "stable optional id",
      "name": "canonical entity name",
      "entity_type": "person|company|project|jurisdiction|policy|concept|risk|document",
      "aliases": [],
      "description": "source-grounded one sentence description",
      "source_ids": ["doc_id"]
    }
  ]
}
```

Rules:
- Prefer canonical names.
- Include aliases only when present in source material.
- Do not extract generic topic words without identity.

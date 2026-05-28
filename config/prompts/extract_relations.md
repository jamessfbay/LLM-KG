# Extract Relations

Convert supported claims into typed subject-predicate-object relations as strict JSON.

Output shape:

```json
{
  "relations": [
    {
      "id": "stable optional id",
      "subject_id": "entity id",
      "predicate": "typed_snake_case_predicate",
      "object_id": "entity id",
      "claim_ids": ["claim_id"],
      "evidence_ids": ["evidence_id"],
      "confidence": 0.0
    }
  ]
}
```

Rules:
- Every relation must be grounded in a claim and evidence.
- Prefer specific predicates such as `affects`, `requires`, `located_on`, `governed_by`, `supports`.
- Avoid generic predicates like `related_to` unless no stronger relation is supported.

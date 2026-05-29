# Extract Relations

Create typed relations only when they are useful for source-linked planning reasoning.

Preferred relation predicates:
- `located_at`, `located_on`, `submitted_to`, `prepared_by`, `owned_by`, `proposed_by`
- `governed_by`, `requires`, `affects`, `supports`, `contradicts`, `part_of`, `mentions`
- `has_issue`, `has_source_gap`, `has_entitlement_scope`, `references_policy`, `changes_site_feature`

Rules:
- No source trace, no relation.
- Every relation must reference existing `subject_id` and `object_id` entity records.
- Every relation must include at least one `claim_ids` or `evidence_ids` value.
- Do not create relations for sheet numbers, drawing labels, or generic plan annotations unless they support a planning fact.
- Avoid `related_to` unless the source shows relevance but no stronger predicate is supported.
- Include a short rationale grounded in the quote that supports the relation.

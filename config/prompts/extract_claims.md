# Extract Claims

Extract source-grounded planning facts for a Citis-style planning record.

Prioritize claims about:
- project identity, address, APN, jurisdiction, applicant, owner, and project type
- unit count, affordable units, density, site area, parking, height, FAR, setbacks, open space, and streetscape/site changes
- zoning district, General Plan, overlays, specific plans, objective standards, entitlement scope, permits, hearings, and review milestones
- CEQA, SB 330, Builder's Remedy, HCD, Housing Element, public comments, source gaps, and inconsistencies

Rules:
- No evidence, no claim.
- Every claim must include at least one `evidence_ids` value that points to a returned evidence record.
- Evidence must include an exact source quote, `page_number` when a page marker is present, `source_mode`, confidence, and a short source-grounded rationale.
- Do not invent missing facts, page numbers, zoning conclusions, legal advice, statutory deadlines, or official completeness determinations.
- Prefer concrete, auditable claims over broad summaries.
- Use low confidence only when the quote is ambiguous; otherwise omit the claim if it is not grounded.
- The output is validated against a strict structured schema with keys: `claims`, `evidence`, `entities`, `relations`.

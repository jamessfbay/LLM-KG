# Extract Entities

Extract canonical entities that are useful for planning reasoning.

Good entity candidates:
- project, address/site, parcel/APN, applicant, owner, architect, organization, agency, jurisdiction
- document, plan set, staff report, policy, zoning code, General Plan, specific plan, overlay, statute, program
- concept, issue, source gap, event, entitlement, permit, public comment theme

Entity rules:
- Use canonical names, not sheet labels, drawing callouts, scale notes, abbreviations, isolated all-caps fragments, or generic construction terms.
- Keep entity names concise and human-readable.
- Include aliases only when the source explicitly shows alternate names.
- Use generic entity types such as `project`, `document`, `policy`, `organization`, `person`, `jurisdiction`, `parcel`, `address`, `event`, `issue`, or `concept`.
- Unknown but potentially important types may be emitted, but they should be conservative and source-grounded.
- Every entity must cite the source document id in `source_ids`.

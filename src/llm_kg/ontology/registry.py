from __future__ import annotations

from dataclasses import dataclass

from llm_kg.models import Entity, OntologySchema, Relation


DEFAULT_ENTITY_TYPES = [
    "concept",
    "organization",
    "person",
    "project",
    "document",
    "policy",
    "event",
    "jurisdiction",
    "risk",
    "location",
]

DEFAULT_RELATION_PREDICATES = [
    "related_to",
    "supports",
    "contradicts",
    "requires",
    "required",
    "limits",
    "limited",
    "affects",
    "affected_by",
    "part_of",
    "mentions",
    "updates",
    "opposes",
    "constrains",
    "enables",
    "is",
]


@dataclass(frozen=True)
class OntologyIssue:
    code: str
    target_type: str
    target_id: str
    message: str


class OntologyRegistry:
    def __init__(self, schema: OntologySchema) -> None:
        self.schema = schema
        self.entity_types = {item.lower() for item in schema.entity_types}
        self.predicates = {item.lower() for item in schema.relation_predicates}

    def validate_entity(self, entity: Entity) -> list[OntologyIssue]:
        if entity.entity_type.lower() in self.entity_types:
            return []
        return [
            OntologyIssue(
                code="unknown_entity_type",
                target_type="entity",
                target_id=entity.id,
                message=f"Unknown entity type: {entity.entity_type}",
            )
        ]

    def validate_relation(self, relation: Relation, entity_ids: set[str]) -> list[OntologyIssue]:
        issues: list[OntologyIssue] = []
        if relation.predicate.lower() not in self.predicates:
            issues.append(
                OntologyIssue(
                    code="unknown_relation_predicate",
                    target_type="relation",
                    target_id=relation.id,
                    message=f"Unknown relation predicate: {relation.predicate}",
                )
            )
        if relation.subject_id not in entity_ids:
            issues.append(
                OntologyIssue(
                    code="relation_missing_subject",
                    target_type="relation",
                    target_id=relation.id,
                    message=f"Relation subject does not exist: {relation.subject_id}",
                )
            )
        if relation.object_id not in entity_ids:
            issues.append(
                OntologyIssue(
                    code="relation_missing_object",
                    target_type="relation",
                    target_id=relation.id,
                    message=f"Relation object does not exist: {relation.object_id}",
                )
            )
        return issues


def build_ontology_registry(profile: str = "generic") -> OntologyRegistry:
    if profile != "generic":
        schema = OntologySchema(id=profile, entity_types=DEFAULT_ENTITY_TYPES, relation_predicates=DEFAULT_RELATION_PREDICATES)
    else:
        schema = OntologySchema(
            id="generic",
            entity_types=DEFAULT_ENTITY_TYPES,
            relation_predicates=DEFAULT_RELATION_PREDICATES,
        )
    return OntologyRegistry(schema)

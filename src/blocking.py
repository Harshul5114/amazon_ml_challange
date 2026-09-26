"""Modular inverted-index candidate retrieval over conservatively normalized text."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from src.normalize import normalize_record


MAX_S1_POSTING = 64
MAX_NAME_TOKEN_DF = 5_000
MAX_RARE_NAME_TOKENS = 2
MAX_ADDRESS_NUMBERS = 3


@dataclass(slots=True)
class Business:
    entity_id: str
    country: str
    business_name: str
    business_address: str
    name_norm: str
    address_norm: str
    name_tokens: tuple[str, ...]
    address_numeric_tokens: tuple[str, ...]

    @classmethod
    def from_raw(cls, row: Mapping[str, str | None]) -> "Business":
        derived = normalize_record(row)
        return cls(
            entity_id=derived["entity_id"],
            country=derived["country"] or "",
            business_name=derived["business_name"] or "",
            business_address=derived["business_address"] or "",
            name_norm=derived["name_norm"],
            address_norm=derived["address_norm"],
            name_tokens=tuple(derived["name_tokens"]),
            address_numeric_tokens=tuple(derived["address_numeric_tokens"]),
        )


KeyFunction = Callable[[Business], Iterable[Hashable]]


@dataclass(slots=True)
class InvertedRule:
    name: str
    s1_keys: KeyFunction
    target_keys: KeyFunction
    max_s1_posting: int | None = None
    index: dict[Hashable, list[int]] = field(default_factory=lambda: defaultdict(list))
    skipped_postings: int = 0

    def add_s1(self, number: int, record: Business) -> None:
        for key in dict.fromkeys(self.s1_keys(record)):
            self.index[key].append(number)

    def retrieve(self, target: Business) -> set[int]:
        found = set()
        for key in dict.fromkeys(self.target_keys(target)):
            posting = self.index.get(key, ())
            if self.max_s1_posting is not None and len(posting) > self.max_s1_posting:
                self.skipped_postings += 1
                continue
            found.update(posting)
        return found


def exact_name_keys(record: Business) -> tuple[str, ...]:
    return (record.name_norm,) if record.name_norm else ()


def name_signature_keys(record: Business) -> tuple[tuple[str, ...], ...]:
    """Sorted token multiset: order-insensitive but retains repeated tokens."""
    return (tuple(sorted(record.name_tokens)),) if len(record.name_tokens) >= 2 else ()


def address_exact_numeric_keys(record: Business) -> tuple[tuple[str, str], ...]:
    """Exact address with a substantial number, within the same country."""
    if record.address_norm and any(len(number) >= 3 for number in record.address_numeric_tokens):
        return ((record.country, record.address_norm),)
    return ()


def address_numbers(record: Business) -> tuple[str, ...]:
    numbers = {number for number in record.address_numeric_tokens if len(number) >= 3}
    return tuple(sorted(numbers, key=lambda number: (-len(number), number))[:MAX_ADDRESS_NUMBERS])


def build_rules(s1_records: Sequence[Business]) -> list[InvertedRule]:
    """Build rules on the S1 query set; target records can then be streamed."""
    token_df = Counter(
        token
        for record in s1_records
        for token in set(record.name_tokens)
        if len(token) >= 3
    )

    def eligible_tokens(record: Business) -> list[str]:
        return [token for token in set(record.name_tokens) if len(token) >= 3 and 0 < token_df[token] <= MAX_NAME_TOKEN_DF]

    def numeric_name_s1_keys(record: Business) -> Iterable[tuple[str, str, str]]:
        tokens = sorted(eligible_tokens(record), key=lambda token: (token_df[token], -len(token), token))[:MAX_RARE_NAME_TOKENS]
        return ((record.country, number, token) for number in address_numbers(record) for token in tokens)

    def numeric_name_target_keys(record: Business) -> Iterable[tuple[str, str, str]]:
        return ((record.country, number, token) for number in address_numbers(record) for token in eligible_tokens(record))

    rules = [
        InvertedRule("A_exact_name", exact_name_keys, exact_name_keys),
        InvertedRule("B_name_signature", name_signature_keys, name_signature_keys),
        InvertedRule("C1_exact_numeric_address", address_exact_numeric_keys, address_exact_numeric_keys, MAX_S1_POSTING),
        InvertedRule("C2_numeric_plus_name_token", numeric_name_s1_keys, numeric_name_target_keys, MAX_S1_POSTING),
    ]
    for number, record in enumerate(s1_records):
        for rule in rules:
            rule.add_s1(number, record)
    return rules


def retrieve_by_rule(rules: Sequence[InvertedRule], target: Business) -> dict[str, set[int]]:
    """Return one S1 candidate set per rule; callers can union any subset."""
    return {rule.name: rule.retrieve(target) for rule in rules}

"""Declarative question primitives with identity hashing and a jaggedness linter.

Questions are declared once, reused across call sites, and identified by a hash
of their spec. Two call sites that declare the same question collapse into a
single question on the wire.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from typesafe_sdk import Choice as SDKChoice
from typesafe_sdk import Noul as SDKNoul
from typesafe_sdk import Score as SDKScore

from .state import canonical, estimate_tokens

# Documented primitive bounds (docs.typesafe.ai/primitives).
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

Severity = Literal["error", "warning"]


class QuestionError(ValueError):
    """Raised when a question violates a hard Jev constraint."""


@dataclass(frozen=True)
class Lint:
    """One advisory finding against a question's wording."""

    code: str
    severity: Severity
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.severity}: {self.code}] {self.message}"


# Each rule targets a failure mode documented in the jev-1.13 jaggedness notes.
# Patterns are intentionally narrow: a linter that cries wolf gets switched off.
_RULES: tuple[tuple[str, Severity, re.Pattern[str], str], ...] = (
    (
        "counting",
        "error",
        re.compile(r"\b(how many|count|number of|tally|total|sum of|average)\b", re.I),
        "Jev recognizes the shape of an answer rather than tallying, and the "
        "error grows with list size. Iterate in code with one question per item.",
    ),
    (
        "arithmetic",
        "error",
        re.compile(r"\b(calculate|compute|multiply|divide|subtract|percentage of)\b", re.I),
        "Jev is not a calculator. Keep arithmetic in code.",
    ),
    (
        "date-comparison",
        "error",
        re.compile(
            r"\b(before or after|comes? (?:first|before|after)|how (?:long|far) (?:apart|between)"
            r"|within \d|older than|more recent than|date range)\b",
            re.I,
        ),
        "Jev reads dates as text, not ordered quantities. Extract components as "
        "choices and do ordering and windowing in code.",
    ),
    (
        "generation",
        "error",
        re.compile(r"\b(write|summariz|generat|rewrite|draft|compose|translate|explain why)\w*\b", re.I),
        "Jev does not generate text. Extract candidates elsewhere and let Jev "
        "pick or score among them.",
    ),
    (
        "low-level-numerics",
        "warning",
        re.compile(r"\b(hex|hexadecimal|rgb|base64|byte|checksum|utf-?8)\b", re.I),
        "Jev performs poorly on hex values, RGB triples and other low-level "
        "encodings. Prefer a semantic description of the value.",
    ),
    (
        "indirection",
        "warning",
        re.compile(r"\b(corresponding|respective|associated|aforementioned|its own)\b", re.I),
        "Multi-hop 'property of a property' reasoning reduces accuracy. Split "
        "into separate questions and join in code.",
    ),
)

_NEGATION = re.compile(r"\b(not|never|without|except|unless|neither|nor|cannot)\b|n't\b", re.I)


def _text_of(value: Any) -> str:
    """Flatten instructions/criteria of any shape into lintable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return canonical(value)


def _lint_text(text: str) -> list[Lint]:
    findings: list[Lint] = []
    for code, severity, pattern, message in _RULES:
        if pattern.search(text):
            findings.append(Lint(code=code, severity=severity, message=message))
    negations = _NEGATION.findall(text)
    if len(negations) >= 2:
        findings.append(
            Lint(
                code="double-negative",
                severity="warning",
                message=(
                    "Double negatives and stacked scoping words are read at face "
                    "value and reduce accuracy. Restate positively."
                ),
            )
        )
    return findings


@dataclass(frozen=True, eq=False)
class Question:
    """Base class for a declared question. Compared and hashed by spec."""

    kind: Literal["noul", "choice", "score"]
    instructions: Any
    criteria: Any = None
    name: str | None = None
    weight: float = 1.0
    _key: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        spec = canonical(
            {"kind": self.kind, "instructions": self.instructions, "criteria": self.criteria}
        )
        digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "_key", digest)

    @property
    def key(self) -> str:
        """Content hash of the question spec; the unit of dedupe and caching."""
        return self._key

    @property
    def label(self) -> str:
        """Human-readable identifier, used for wire keys and telemetry."""
        return self.name or f"{self.kind}_{self._key[:8]}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Question) and other._key == self._key

    def __hash__(self) -> int:
        return hash(self._key)

    def lint(self) -> list[Lint]:
        """Advisory findings against this question's wording."""
        return _lint_text(f"{_text_of(self.instructions)} {_text_of(self.criteria)}")

    def estimate_tokens(self) -> int:
        return estimate_tokens(
            {"instructions": self.instructions, "criteria": self.criteria}
        )

    def to_sdk(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class NoulQuestion(Question):
    def to_sdk(self) -> SDKNoul:
        if self.criteria is None:
            return SDKNoul(instructions=self.instructions)
        return SDKNoul(instructions=self.instructions, criteria=self.criteria)


@dataclass(frozen=True, eq=False)
class ChoiceQuestion(Question):
    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.criteria, Mapping) or not self.criteria:
            raise QuestionError("choice() requires a non-empty mapping of options")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise QuestionError(
                f"choice() accepts at most {MAX_CHOICE_OPTIONS} options, "
                f"got {len(self.criteria)}"
            )

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(self.criteria)

    def to_sdk(self) -> SDKChoice:
        return SDKChoice(instructions=self.instructions, criteria=dict(self.criteria))


@dataclass(frozen=True, eq=False)
class ScoreQuestion(Question):
    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.criteria, (str, bytes)) or not isinstance(self.criteria, Sequence):
            raise QuestionError("score() requires an ordered sequence of levels")
        if not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
            raise QuestionError(
                f"score() requires {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} ordered "
                f"levels, got {len(self.criteria)}"
            )

    @property
    def levels(self) -> int:
        return len(self.criteria)

    def to_sdk(self) -> SDKScore:
        return SDKScore(instructions=self.instructions, criteria=list(self.criteria))


def noul(instructions: Any, *, criteria: Any = None, name: str | None = None) -> NoulQuestion:
    """Declare a statement whose truth Jev estimates as a 0-1 probability."""
    return NoulQuestion(kind="noul", instructions=instructions, criteria=criteria, name=name)


def choice(
    instructions: Any, criteria: Mapping[str, Any], *, name: str | None = None
) -> ChoiceQuestion:
    """Declare a selection over a labelled set of mutually exclusive options."""
    return ChoiceQuestion(kind="choice", instructions=instructions, criteria=criteria, name=name)


def score(
    instructions: Any, criteria: Sequence[Any], *, name: str | None = None, weight: float = 1.0
) -> ScoreQuestion:
    """Declare a position on an ordered spectrum of 2-10 levels."""
    return ScoreQuestion(
        kind="score", instructions=instructions, criteria=criteria, name=name, weight=weight
    )


def lint_all(questions: Iterable[Question]) -> dict[str, list[Lint]]:
    """Lint a collection of questions, keyed by label; empty results omitted."""
    return {q.label: findings for q in questions if (findings := q.lint())}

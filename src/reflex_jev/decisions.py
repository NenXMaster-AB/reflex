"""Typed decision wrappers and confidence-gated control flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .questions import Question


class Gate(str, Enum):
    """What a caller is cleared to do with a decision at its confidence."""

    ACT = "act"
    CONFIRM = "confirm"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Policy:
    """Confidence thresholds for one class of action.

    Thresholds are not universal: the same system should gate a read-only
    lookup and an irreversible write at different confidences. Construct these
    per action, not per application.
    """

    act: float = 0.85
    confirm: float = 0.5
    name: str = "default"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confirm <= self.act <= 1.0:
            raise ValueError(
                f"policy requires 0 <= confirm <= act <= 1; "
                f"got confirm={self.confirm}, act={self.act}"
            )

    def gate(self, confidence: float) -> Gate:
        if confidence >= self.act:
            return Gate.ACT
        if confidence >= self.confirm:
            return Gate.CONFIRM
        return Gate.ESCALATE


#: Read-only or trivially reversible actions.
READ_ONLY = Policy(act=0.6, confirm=0.35, name="read_only")
#: The documented starting point for ordinary automation.
STANDARD = Policy(act=0.85, confirm=0.5, name="standard")
#: Irreversible, outward-facing or costly actions.
DESTRUCTIVE = Policy(act=0.95, confirm=0.8, name="destructive")


@dataclass(frozen=True)
class Decision:
    """Base class for one answered question."""

    question: Question
    confidence: float
    raw: Any = field(repr=False, default=None)
    cached: bool = False

    @property
    def label(self) -> str:
        return self.question.label

    def gate(self, policy: Policy = STANDARD) -> Gate:
        """Classify this decision as act / confirm / escalate."""
        return policy.gate(self.confidence)

    def acts(self, policy: Policy = STANDARD) -> bool:
        """True only when confidence clears the policy's automatic-action bar."""
        return self.gate(policy) is Gate.ACT


@dataclass(frozen=True)
class NoulDecision(Decision):
    """A truth estimate for a statement, as a probability in [0, 1].

    Jev returns no confidence for nouls, so ``confidence`` here is *derived* by
    this library as ``|p - 0.5| * 2`` -- distance from maximal uncertainty. It
    is not the model's own calibrated confidence signal, and it is not
    comparable across question types. Note also that Jev guarantees no
    relationship between ``P(s)`` and ``P(not s)``, so do not treat nouls as a
    boolean algebra: ask the question you actually mean.
    """

    p: float = 0.0
    threshold: float = 0.5

    @property
    def value(self) -> bool:
        return self.p >= self.threshold

    def __bool__(self) -> bool:
        return self.value

    def at(self, threshold: float) -> bool:
        """Evaluate truth at an explicit threshold instead of the default."""
        return self.p >= threshold


@dataclass(frozen=True)
class ChoiceDecision(Decision):
    """A selection among labelled options, with the full distribution."""

    value: str = ""
    probabilities: Mapping[str, float] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Compare against the chosen option label for ergonomic branching."""
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.question.key, self.value))

    def ranked(self) -> list[tuple[str, float]]:
        """Options from most to least probable."""
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)

    def runner_up(self) -> tuple[str, float] | None:
        """The second-place option, useful for margin-based routing."""
        ranking = self.ranked()
        return ranking[1] if len(ranking) > 1 else None

    def margin(self) -> float:
        """Probability gap between the top two options."""
        ranking = self.ranked()
        if len(ranking) < 2:
            return ranking[0][1] if ranking else 0.0
        return ranking[0][1] - ranking[1][1]


@dataclass(frozen=True)
class ScoreDecision(Decision):
    """A position on an ordered spectrum.

    ``value`` interpolates between levels (e.g. ``1.035``), but Jev cannot
    reconstruct exact quantities between levels -- treat it as an ordering
    signal, not a measurement.
    """

    value: float = 0.0
    probabilities: Mapping[int, float] = field(default_factory=dict)
    legend: Mapping[int, Any] = field(default_factory=dict)

    @property
    def level(self) -> int:
        """Nearest discrete level."""
        return int(round(self.value))

    @property
    def normalized(self) -> float:
        """Position in [0, 1] across the spectrum, for combining scores."""
        top = max(self.legend) if self.legend else max(0, len(self.probabilities) - 1)
        return self.value / top if top else 0.0

    def describe(self) -> Any:
        """The legend entry for the nearest level."""
        return self.legend.get(self.level)


def composite(
    decisions: Iterable[ScoreDecision],
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Combine several score dimensions into one normalized [0, 1] value.

    Weights default to each question's declared ``weight``. Combining happens
    here, in code, rather than by asking Jev one broad question -- decomposing
    into atomic questions and joining the results is the documented way to get
    reliable answers out of a System One model.
    """
    items = list(decisions)
    if not items:
        return 0.0
    ws = list(weights) if weights is not None else [d.question.weight for d in items]
    if len(ws) != len(items):
        raise ValueError(f"expected {len(items)} weights, got {len(ws)}")
    total = sum(ws)
    if total == 0:
        return 0.0
    return sum(d.normalized * w for d, w in zip(items, ws)) / total

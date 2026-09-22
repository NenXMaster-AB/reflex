"""Decision typing, gating and composition."""

from __future__ import annotations

import pytest

from reflex_jev import (
    DESTRUCTIVE,
    READ_ONLY,
    STANDARD,
    ChoiceDecision,
    Gate,
    NoulDecision,
    Policy,
    ScoreDecision,
    choice,
    composite,
    noul,
    score,
)

REFUND = noul("The customer is explicitly asking for a refund", name="refund")


async def test_answers_map_to_typed_decisions(rx):
    answers = await rx.ask_all(
        {"t": 1},
        c=choice("pick", {"a": "A", "b": "B"}),
        s=score("rate", ["low", "mid", "high"]),
        n=noul("true?"),
    )
    assert isinstance(answers["c"], ChoiceDecision)
    assert isinstance(answers["s"], ScoreDecision)
    assert isinstance(answers["n"], NoulDecision)
    assert answers["s"].legend == {0: "low", 1: "mid", 2: "high"}
    assert answers["s"].describe() == "mid"


async def test_noul_confidence_is_derived_from_distance_to_half(rx, fake):
    fake.noul_p = 0.5
    undecided = await rx.ask({"t": 1}, REFUND)
    assert undecided.confidence == pytest.approx(0.0)
    assert undecided.gate() is Gate.ESCALATE

    certain = await rx.ask({"t": 2}, noul("something else"))
    assert certain.p == pytest.approx(0.5)


def test_noul_thresholding():
    d = NoulDecision(question=REFUND, confidence=0.4, p=0.7)
    assert bool(d) is True
    assert d.at(0.8) is False
    assert d.at(0.6) is True


def test_gates_scale_with_risk():
    d = NoulDecision(question=REFUND, confidence=0.7, p=0.85)
    assert d.gate(READ_ONLY) is Gate.ACT
    assert d.gate(STANDARD) is Gate.CONFIRM
    assert d.gate(DESTRUCTIVE) is Gate.ESCALATE
    assert d.acts(READ_ONLY) and not d.acts(DESTRUCTIVE)


def test_policy_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="confirm <= act"):
        Policy(act=0.4, confirm=0.9)


def test_choice_compares_to_its_label_and_exposes_margin():
    q = choice("pick", {"a": "A", "b": "B", "c": "C"})
    d = ChoiceDecision(
        question=q, confidence=0.6, value="a", probabilities={"a": 0.5, "b": 0.4, "c": 0.1}
    )
    assert d == "a"
    assert d != "b"
    assert d.ranked()[0] == ("a", 0.5)
    assert d.runner_up() == ("b", 0.4)
    assert d.margin() == pytest.approx(0.1)


def test_score_normalizes_across_its_own_spectrum():
    q = score("rate", ["a", "b", "c", "d", "e"])
    d = ScoreDecision(
        question=q, confidence=0.8, value=2.0, probabilities={i: 0.2 for i in range(5)},
        legend={i: x for i, x in enumerate("abcde")},
    )
    assert d.level == 2
    assert d.normalized == pytest.approx(0.5)


def test_composite_weights_dimensions():
    q_low = score("a", ["x", "y", "z"], weight=1.0)
    q_high = score("b", ["x", "y", "z"], weight=3.0)
    low = ScoreDecision(question=q_low, confidence=1.0, value=0.0, legend={0: "x", 1: "y", 2: "z"})
    high = ScoreDecision(question=q_high, confidence=1.0, value=2.0, legend={0: "x", 1: "y", 2: "z"})
    assert composite([low, high]) == pytest.approx(0.75)
    assert composite([low, high], weights=[1, 1]) == pytest.approx(0.5)
    assert composite([]) == 0.0
    with pytest.raises(ValueError, match="expected 2 weights"):
        composite([low, high], weights=[1])

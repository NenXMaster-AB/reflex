"""A fake Jev client that records requests, so batching can be asserted on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

from reflex_jev import Reflex


@dataclass
class RecordedCall:
    state: Any
    questions: dict[str, Any]
    model: str | None


class FakeJev:
    """Stands in for AsyncTypeSafeClient, answering deterministically."""

    def __init__(self, *, noul_p: float = 0.8, fail: Exception | None = None,
                 omit: set[str] | None = None) -> None:
        self.calls: list[RecordedCall] = []
        self.noul_p = noul_p
        self.fail = fail
        self.omit = omit or set()
        self.closed = False

    @property
    def request_count(self) -> int:
        return len(self.calls)

    @property
    def questions_sent(self) -> int:
        return sum(len(c.questions) for c in self.calls)

    async def system_one(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None, **kw: Any
    ) -> SystemOneResponse:
        self.calls.append(RecordedCall(state=state, questions=dict(questions), model=model))
        if self.fail is not None:
            raise self.fail
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            if key in self.omit:
                continue
            answers[key] = self._answer(question)
        return SystemOneResponse(
            model=model or "fake-jev",
            usage=Usage(input_tokens=100, output_tokens=0),
            answers=answers,
        )

    def _answer(self, question: Any) -> Any:
        if isinstance(question, Noul):
            return NoulAnswer(noul=self.noul_p)
        if isinstance(question, Choice):
            options = list(question.criteria)
            top = options[0]
            rest = (1.0 - 0.9) / max(1, len(options) - 1)
            probs = {o: (0.9 if o == top else rest) for o in options}
            return ChoiceAnswer(choice=top, confidence=0.9, probabilities=probs)
        if isinstance(question, Score):
            levels = list(question.criteria)
            n = len(levels)
            probs = {i: (0.8 if i == 1 else 0.2 / max(1, n - 1)) for i in range(n)}
            legend = {i: levels[i] for i in range(n)}
            return ScoreAnswer(score=1.0, confidence=0.75, legend=legend, probabilities=probs)
        raise AssertionError(f"unexpected question type {type(question)}")

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake() -> FakeJev:
    return FakeJev()


@pytest.fixture
def rx(fake: FakeJev) -> Reflex:
    return Reflex(client=fake, model="jev-test")

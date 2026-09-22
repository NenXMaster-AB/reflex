"""The batching scheduler.

Jev evaluates every question in a request in parallel against one shared state,
charges only for input tokens, and gives output away free. So the cost of a
question is dominated by the state it rides along with -- asking five questions
about one state in one request is close to the price of asking one, while
asking them in five requests costs five times as much.

This scheduler makes that the default. Questions submitted against identical
state within a single event-loop tick are coalesced into one request,
deduplicated by question identity, and served from cache where possible. Callers
write ordinary straight-line code and get the batched request for free.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from .cache import Cache, cache_key
from .decisions import ChoiceDecision, Decision, NoulDecision, ScoreDecision
from .errors import AnswerMissing, BudgetError
from .questions import Question
from .state import (
    MAX_STATE_PLUS_LONGEST_QUESTION_TOKENS,
    MAX_TOTAL_TOKENS,
    estimate_tokens,
)


def wire_key(question: Question) -> str:
    """Collision-free key for a question on the wire.

    Derived from the spec hash rather than the user-facing name, so two
    differently-named but identical questions collapse, and two identically
    named but different questions do not collide.
    """
    return f"q_{question.key}"


@dataclass
class RequestStats:
    """One flushed request, for telemetry and cost accounting."""

    state_key: str
    model: str
    questions: int
    deduped: int
    input_tokens: int | None
    latency_ms: float
    chunks: int


@dataclass
class _Batch:
    """Questions accumulated against one state, awaiting flush."""

    state: Any
    state_key: str
    model: str
    questions: dict[str, Question] = field(default_factory=dict)
    waiters: dict[str, list[asyncio.Future]] = field(default_factory=dict)
    submitted: int = 0
    scheduled: bool = False

    def add(self, question: Question, future: asyncio.Future) -> None:
        self.submitted += 1
        self.questions.setdefault(question.key, question)
        self.waiters.setdefault(question.key, []).append(future)


def plan_chunks(
    state_tokens: int, questions: Sequence[Question]
) -> list[list[Question]]:
    """Pack questions into requests that respect Jev's documented limits.

    Two limits apply at once: 64k tokens for state plus all questions, and 32k
    for state plus the single longest question. Packing is greedy in declaration
    order, which keeps chunk membership stable and therefore cacheable.
    """
    if state_tokens >= MAX_STATE_PLUS_LONGEST_QUESTION_TOKENS:
        raise BudgetError(
            f"state is ~{state_tokens} tokens, which leaves no room under the "
            f"{MAX_STATE_PLUS_LONGEST_QUESTION_TOKENS}-token state-plus-question "
            "limit. Retrieve and filter the state before sending it: Jev's "
            "accuracy also degrades when state carries irrelevant material."
        )

    chunks: list[list[Question]] = []
    current: list[Question] = []
    current_sum = 0

    for question in questions:
        size = question.estimate_tokens()
        if state_tokens + size > MAX_STATE_PLUS_LONGEST_QUESTION_TOKENS:
            raise BudgetError(
                f"question {question.label!r} (~{size} tokens) plus state "
                f"(~{state_tokens} tokens) exceeds the "
                f"{MAX_STATE_PLUS_LONGEST_QUESTION_TOKENS}-token limit"
            )
        if current and state_tokens + current_sum + size > MAX_TOTAL_TOKENS:
            chunks.append(current)
            current, current_sum = [], 0
        current.append(question)
        current_sum += size

    if current:
        chunks.append(current)
    return chunks


def to_decision(question: Question, answer: Any, *, cached: bool = False) -> Decision:
    """Convert an SDK answer into the matching typed decision."""
    if isinstance(answer, NoulAnswer):
        p = float(answer.noul)
        # Jev returns no confidence for nouls. Derive one as distance from
        # maximal uncertainty, and label it as ours in NoulDecision's docstring.
        return NoulDecision(
            question=question,
            confidence=abs(p - 0.5) * 2.0,
            raw=answer,
            cached=cached,
            p=p,
        )
    if isinstance(answer, ChoiceAnswer):
        return ChoiceDecision(
            question=question,
            confidence=float(answer.confidence),
            raw=answer,
            cached=cached,
            value=answer.choice,
            probabilities=dict(answer.probabilities),
        )
    if isinstance(answer, ScoreAnswer):
        return ScoreDecision(
            question=question,
            confidence=float(answer.confidence),
            raw=answer,
            cached=cached,
            value=float(answer.score),
            probabilities=dict(answer.probabilities),
            legend=dict(answer.legend),
        )
    raise TypeError(f"unrecognized answer type {type(answer).__name__}")


class Scheduler:
    """Coalesces question submissions into batched Jev requests."""

    def __init__(
        self,
        client: Any,
        *,
        model: str | None,
        cache: Cache,
        on_request: Callable[[RequestStats], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._cache = cache
        self._on_request = on_request
        self._batches: dict[tuple[str, str], _Batch] = {}
        self._inflight: set[asyncio.Task] = set()

    # -- submission ------------------------------------------------------

    def submit(
        self,
        state: Any,
        state_key: str,
        question: Question,
        *,
        model: str | None = None,
        defer: bool = False,
    ) -> asyncio.Future:
        """Register a question and return a future for its decision.

        Synchronous by design: it must be possible to register many questions
        without yielding to the event loop, otherwise each one would flush
        separately and the batching would be defeated.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        effective_model = model or self._model or "jev-latest"

        hit = self._cache.get(cache_key(state_key, question.key, effective_model))
        if hit is not None:
            future.set_result(to_decision(question, hit, cached=True))
            return future

        batch_id = (state_key, effective_model)
        batch = self._batches.get(batch_id)
        if batch is None:
            batch = _Batch(state=state, state_key=state_key, model=effective_model)
            self._batches[batch_id] = batch
        batch.add(question, future)

        if not defer and not batch.scheduled:
            batch.scheduled = True
            loop.call_soon(self._start_flush, batch_id)
        return future

    # -- flushing --------------------------------------------------------

    def _start_flush(self, batch_id: tuple[str, str]) -> None:
        batch = self._batches.pop(batch_id, None)
        if batch is None or not batch.questions:
            return
        task = asyncio.ensure_future(self._flush(batch))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    def flush_now(self, batch_id: tuple[str, str]) -> asyncio.Task | None:
        """Flush one pending batch immediately; used by explicit batch scopes."""
        batch = self._batches.pop(batch_id, None)
        if batch is None or not batch.questions:
            return None
        task = asyncio.ensure_future(self._flush(batch))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        return task

    async def _flush(self, batch: _Batch) -> None:
        questions = list(batch.questions.values())
        futures = batch.waiters
        try:
            state_tokens = estimate_tokens(batch.state)
            chunks = plan_chunks(state_tokens, questions)
            started = time.perf_counter()

            results = await asyncio.gather(
                *(self._send(batch, chunk) for chunk in chunks)
            )

            answers: dict[str, Any] = {}
            input_tokens = 0
            saw_usage = False
            for chunk_answers, usage in results:
                answers.update(chunk_answers)
                if usage is not None and usage.input_tokens is not None:
                    input_tokens += usage.input_tokens
                    saw_usage = True

            latency_ms = (time.perf_counter() - started) * 1000

            for question in questions:
                answer = answers.get(wire_key(question))
                waiting = futures.get(question.key, [])
                if answer is None:
                    error = AnswerMissing(
                        f"Jev returned no answer for question {question.label!r}"
                    )
                    for future in waiting:
                        if not future.done():
                            future.set_exception(error)
                    continue
                self._cache.set(
                    cache_key(batch.state_key, question.key, batch.model), answer
                )
                decision = to_decision(question, answer)
                for future in waiting:
                    if not future.done():
                        future.set_result(decision)

            if self._on_request is not None:
                self._on_request(
                    RequestStats(
                        state_key=batch.state_key,
                        model=batch.model,
                        questions=len(questions),
                        deduped=batch.submitted - len(questions),
                        input_tokens=input_tokens if saw_usage else None,
                        latency_ms=latency_ms,
                        chunks=len(chunks),
                    )
                )
        except BaseException as exc:  # noqa: BLE001 - every waiter must be resolved
            for waiting in futures.values():
                for future in waiting:
                    if not future.done():
                        future.set_exception(exc)
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _send(
        self, batch: _Batch, chunk: Sequence[Question]
    ) -> tuple[Mapping[str, Any], Any]:
        payload = {wire_key(q): q.to_sdk() for q in chunk}
        response = await self._client.system_one(
            batch.state, payload, model=batch.model
        )
        return response.answers, getattr(response, "usage", None)

    async def drain(self) -> None:
        """Wait for every in-flight request to settle.

        Tasks are discarded explicitly rather than relying on their done
        callbacks. Those callbacks run via ``call_soon``, but awaiting a gather
        over already-completed tasks returns without yielding to the loop -- so
        a version that waited for the callbacks to empty the set would spin
        forever, never yielding, and could not even be cancelled.
        """
        while self._inflight:
            pending = list(self._inflight)
            await asyncio.gather(*pending, return_exceptions=True)
            self._inflight.difference_update(pending)

    def pending_ids(self) -> Iterable[tuple[str, str]]:
        return list(self._batches)

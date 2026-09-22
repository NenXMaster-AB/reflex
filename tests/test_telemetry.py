"""The on_request hook, which is how cost and batching get observed in prod."""

from __future__ import annotations

import asyncio

import pytest

from reflex_jev import Reflex, RequestStats, noul


async def test_stats_report_dedupe_and_usage(fake):
    seen: list[RequestStats] = []
    rx = Reflex(client=fake, model="jev-test", on_request=seen.append)

    view = rx.view({"t": 1})
    q = noul("The message conveys urgency")
    await asyncio.gather(view.ask(q), view.ask(q), view.ask(noul("Another statement")))

    assert len(seen) == 1
    stats = seen[0]
    assert stats.questions == 2, "two distinct questions reached the wire"
    assert stats.deduped == 1, "one submission was served by dedupe"
    assert stats.chunks == 1
    assert stats.input_tokens == 100
    assert stats.model == "jev-test"
    assert stats.latency_ms >= 0


async def test_split_requests_are_reported_as_one_flush(fake):
    seen: list[RequestStats] = []
    rx = Reflex(client=fake, on_request=seen.append)
    view = rx.view({"t": "small"})
    questions = [noul(f"{i} " + "y" * 100_000) for i in range(3)]
    await asyncio.gather(*(view.ask(q) for q in questions))

    assert len(seen) == 1
    assert seen[0].chunks == 2
    assert seen[0].input_tokens == 200, "usage summed across both chunks"


async def test_cached_answers_emit_no_request(fake):
    seen: list[RequestStats] = []
    rx = Reflex(client=fake, on_request=seen.append)
    q = noul("The message conveys urgency")
    await rx.ask({"t": 1}, q)
    await rx.ask({"t": 1}, q)
    assert len(seen) == 1


async def test_drain_returns_promptly_after_work_completes(fake):
    """Regression: drain used to spin without yielding, and could not be cancelled.

    Task done-callbacks run via ``call_soon``, but awaiting a gather over
    already-finished tasks does not yield, so the set never emptied.
    """
    rx = Reflex(client=fake)
    await rx.ask({"t": 1}, noul("a statement"))
    await asyncio.wait_for(rx.drain(), timeout=2)


async def test_aclose_returns_promptly(fake):
    rx = Reflex(client=fake)
    await asyncio.gather(
        rx.ask({"t": 1}, noul("first")), rx.ask({"t": 2}, noul("second"))
    )
    await asyncio.wait_for(rx.aclose(), timeout=2)


async def test_drain_is_a_noop_when_idle(fake):
    rx = Reflex(client=fake)
    await asyncio.wait_for(rx.drain(), timeout=2)

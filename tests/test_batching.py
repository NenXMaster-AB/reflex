"""The central claim: many questions, one state, one request."""

from __future__ import annotations

import asyncio

import pytest

from reflex_jev import Reflex, choice, noul, score, untrusted

TICKET = {"subject": "Duplicate charge", "body": "I was charged twice for A-104."}

DEPARTMENT = choice(
    "Which team should handle this",
    {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems"},
    name="department",
)
FRUSTRATION = score(
    "How frustrated the customer appears",
    ["Calm, just stating facts", "Frustrated but civil", "Very angry"],
    name="frustration",
)
REFUND = noul("The customer is explicitly asking for a refund", name="refund")


async def test_concurrent_asks_collapse_into_one_request(rx, fake):
    view = rx.view(TICKET)
    dept, anger, refund = await asyncio.gather(
        view.ask(DEPARTMENT), view.ask(FRUSTRATION), view.ask(REFUND)
    )
    assert fake.request_count == 1
    assert fake.questions_sent == 3
    assert dept.value == "billing"
    assert anger.level == 1
    assert refund.p == pytest.approx(0.8)


async def test_ask_all_is_one_request(rx, fake):
    answers = await rx.ask_all(TICKET, DEPARTMENT, FRUSTRATION, REFUND)
    assert fake.request_count == 1
    assert set(answers) == {"department", "frustration", "refund"}


async def test_ask_all_accepts_keyword_names(rx, fake):
    answers = await rx.ask_all(TICKET, dept=DEPARTMENT, anger=FRUSTRATION)
    assert fake.request_count == 1
    assert set(answers) == {"dept", "anger"}


async def test_identical_questions_are_deduplicated(rx, fake):
    view = rx.view(TICKET)
    same = noul("The customer is explicitly asking for a refund")
    a, b, c = await asyncio.gather(view.ask(REFUND), view.ask(same), view.ask(REFUND))
    assert fake.request_count == 1
    assert fake.questions_sent == 1, "one question should reach the wire"
    assert a.p == b.p == c.p


async def test_repeat_question_is_served_from_cache(rx, fake):
    await rx.ask(TICKET, REFUND)
    assert fake.request_count == 1
    again = await rx.ask(TICKET, REFUND)
    assert fake.request_count == 1, "cached result must not hit the API"
    assert again.cached is True


async def test_cache_misses_when_state_changes(rx, fake):
    await rx.ask(TICKET, REFUND)
    await rx.ask({**TICKET, "body": "different"}, REFUND)
    assert fake.request_count == 2


async def test_cache_is_insensitive_to_key_order(rx, fake):
    await rx.ask({"a": 1, "b": 2}, REFUND)
    await rx.ask({"b": 2, "a": 1}, REFUND)
    assert fake.request_count == 1, "canonical state hashing should match"


async def test_different_states_do_not_share_a_request(rx, fake):
    a, b = await asyncio.gather(
        rx.view({"x": 1}).ask(REFUND), rx.view({"x": 2}).ask(REFUND)
    )
    assert fake.request_count == 2
    assert a.p == b.p


async def test_sequential_awaits_cost_separate_requests(rx, fake):
    """Documents the limit of automatic batching: it spans one loop tick."""
    view = rx.view(TICKET)
    await view.ask(DEPARTMENT)
    await view.ask(FRUSTRATION)
    assert fake.request_count == 2


async def test_explicit_batch_spans_awaits(rx, fake):
    """The escape hatch for the case above."""
    async with rx.batch(TICKET) as view:
        dept = view.ask(DEPARTMENT)
        await asyncio.sleep(0)  # an await in the middle
        anger = view.ask(FRUSTRATION)
    assert fake.request_count == 1
    assert (await dept).value == "billing"
    assert (await anger).level == 1


async def test_caching_can_be_disabled(fake):
    rx = Reflex(client=fake, cache=False)
    await rx.ask(TICKET, REFUND)
    await rx.ask(TICKET, REFUND)
    assert fake.request_count == 2


async def test_model_is_passed_through(rx, fake):
    await rx.ask(TICKET, REFUND)
    assert fake.calls[0].model == "jev-test"


async def test_concurrent_batch_scopes_do_not_flush_each_other(rx, fake):
    """Exiting one explicit scope must not cut short another that is still open."""
    outer = rx.batch({"s": "outer"})
    outer_view = await outer.__aenter__()
    first = outer_view.ask(DEPARTMENT)

    async with rx.batch({"s": "inner"}) as inner_view:
        inner_view.ask(REFUND)
        inner_view.ask(FRUSTRATION)

    assert fake.request_count == 1, "only the inner scope should have flushed"
    assert fake.questions_sent == 2

    second = outer_view.ask(REFUND)
    await outer.__aexit__(None, None, None)

    assert fake.request_count == 2
    assert (await first).value == "billing"
    assert (await second).p == pytest.approx(0.8)


async def test_view_guard_is_none_before_asking(rx):
    assert rx.view(TICKET).guard is None


async def test_batch_scope_with_only_cache_hits_makes_no_request(rx, fake):
    await rx.ask_all(TICKET, DEPARTMENT, FRUSTRATION)
    assert fake.request_count == 1

    async with rx.batch(TICKET) as view:
        dept = view.ask(DEPARTMENT)
        anger = view.ask(FRUSTRATION)

    assert fake.request_count == 1, "nothing left to flush"
    assert (await dept).cached is True
    assert (await anger).cached is True


async def test_all_inside_a_batch_scope_does_not_deadlock(rx, fake):
    """Regression: all() awaits, so inside a deferred scope it must flush itself."""
    async with rx.batch(TICKET) as view:
        answers = await asyncio.wait_for(
            view.all(DEPARTMENT, FRUSTRATION), timeout=2
        )
    assert set(answers) == {"department", "frustration"}
    assert fake.request_count == 1


async def test_guarded_all_inside_a_batch_scope_does_not_deadlock(rx, fake):
    """The guard rides a different state, so its batch must flush too."""
    fake.noul_p = 0.1
    async with rx.batch({"msg": untrusted("hello")}) as view:
        answers = await asyncio.wait_for(view.all(REFUND), timeout=2)
    assert set(answers) == {"refund"}
    assert view.guard is not None
    assert fake.request_count == 2

"""Live check against the real Jev API.

Runs the same five questions two ways -- batched into one request, and one
request each -- and reports latency and billed input tokens for both.

    export TYPESAFE_API_KEY="sk-..."
    python examples/live_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from reflex_jev import (
    DESTRUCTIVE,
    Gate,
    Reflex,
    RequestStats,
    choice,
    noul,
    score,
    untrusted,
)

TICKET = {
    "ticket": {
        "subject": "Duplicate charge",
        "messages": [
            {
                "from": "customer",
                "text": untrusted(
                    "I was charged twice for order A-104. Please refund the "
                    "duplicate, this is the third time I've written in."
                ),
            }
        ],
    },
    "order": {
        "id": "A-104",
        "charges": [
            {"amount_usd": 49, "status": "captured"},
            {"amount_usd": 49, "status": "captured"},
        ],
    },
    "refund_policy": "Duplicate charges are eligible for a refund.",
}

DEPARTMENT = choice(
    "Which team should handle this",
    {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions",
    },
    name="department",
)
FRUSTRATION = score(
    "How frustrated the customer appears",
    ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
    name="frustration",
)
REFUND_REQUESTED = noul("The customer is explicitly asking for a refund", name="refund_requested")
POLICY_SUPPORTS = noul("The stated refund policy covers this situation", name="policy_supports")
IS_REPEAT = noul("The customer says they have contacted support before", name="is_repeat")

QUESTIONS = [DEPARTMENT, FRUSTRATION, REFUND_REQUESTED, POLICY_SUPPORTS, IS_REPEAT]


async def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set.", file=sys.stderr)
        return 2

    model = os.environ.get("REFLEX_MODEL", "jev-1.13.0")
    batched: list[RequestStats] = []

    print(f"model: {model}\n")

    # --- batched: one request for all five questions ---------------------
    async with Reflex(model=model, on_request=batched.append) as rx:
        view = rx.view(TICKET)
        started = time.perf_counter()
        answers = await view.all(*QUESTIONS)
        batched_ms = (time.perf_counter() - started) * 1000

        print("answers")
        for name, d in answers.items():
            if hasattr(d, "p"):
                detail = f"p={d.p:.3f}"
            elif hasattr(d, "level"):
                detail = f"score={d.value:.3f} level={d.level} ({d.describe()!r})"
            else:
                detail = f"{d.value!r} margin={d.margin():.3f}"
            print(f"  {name:<18} {detail:<40} confidence={d.confidence:.3f}  {d.gate().value}")

        # The guard rides along in the same request because the customer message
        # is wrapped in untrusted().
        print(f"\n  injection guard    p={view.guard.p:.3f}")

        refund = answers["refund_requested"]
        supported = answers["policy_supports"]
        if refund.at(0.7) and supported.at(0.7) and supported.gate(DESTRUCTIVE) is Gate.ACT:
            verdict = "auto-refund"
        elif refund.at(0.5):
            verdict = "queue for human approval"
        else:
            verdict = "no refund action"
        print(f"  routed to          {verdict}\n")

        # --- unbatched: one request per question -------------------------
        separate: list[RequestStats] = []
        rx2 = Reflex(model=model, cache=False, on_request=separate.append)
        started = time.perf_counter()
        for q in QUESTIONS:
            await rx2.ask(TICKET, q)
        separate_ms = (time.perf_counter() - started) * 1000
        await rx2.aclose()

    batched_tokens = sum(s.input_tokens or 0 for s in batched)
    separate_tokens = sum(s.input_tokens or 0 for s in separate)

    print(f"{'':<12}{'requests':>10}{'input tokens':>14}{'wall ms':>10}")
    print(f"{'batched':<12}{len(batched):>10}{batched_tokens:>14}{batched_ms:>10.0f}")
    print(f"{'separate':<12}{len(separate):>10}{separate_tokens:>14}{separate_ms:>10.0f}")
    if batched_tokens:
        print(
            f"\nbatching saved {separate_tokens - batched_tokens} input tokens "
            f"({separate_tokens / batched_tokens:.1f}x) and "
            f"{separate_ms - batched_ms:.0f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

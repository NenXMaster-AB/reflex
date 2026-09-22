# reflex-jev

Batching, caching and confidence-gated control flow for [TypeSafe AI's Jev](https://typesafe.ai),
the System One model that returns typed, calibrated decisions instead of text.

## Why

Jev evaluates every question in a request **in parallel against one shared state**,
bills **input tokens only** ($0.042/MTok), and gives output away free. So the cost of a
question is dominated by the state it rides along with: asking five questions about one
state in one request costs about the same as asking one, while asking them in five
requests costs five times as much.

That makes the naive shape — a semantic check written at each call site, each one
re-sending the same state — quietly expensive. `reflex-jev` makes the economical shape
the default:

```python
import asyncio
from reflex_jev import Reflex, noul, choice, score

department = choice("Which team should handle this", {
    "billing":   "Payment or subscription issues",
    "technical": "Bugs or integration problems",
    "sales":     "Pricing or account questions",
})
frustration = score("How frustrated the customer appears",
                    ["Calm, just stating facts", "Frustrated but civil", "Very angry"])
refund_requested = noul("The customer is explicitly asking for a refund")

async def main():
    async with Reflex() as rx:
        v = rx.view(ticket)
        dept, anger, refund = await asyncio.gather(
            v.ask(department), v.ask(frustration), v.ask(refund_requested),
        )   # one request, not three
        print(dept.value, dept.confidence, anger.level, refund.p)

asyncio.run(main())
```

Three `ask` calls, **one** HTTP request. The scheduler coalesces every question submitted
against identical state within a single event-loop tick — DataLoader semantics, applied to
Jev's economics.

## What it gives you

**Automatic batching.** Questions registered before the next `await` share a request.
`ask()` is deliberately synchronous so you can register many without yielding.

**Dedupe and caching.** Keyed on `(state hash, question spec hash, model)`. Two call sites
asking the same question cost one question on the wire; a repeat costs nothing.

**Speculative fan-out.** `view.all(...)` asks everything up front, including questions you
may not need, and you branch in code:

```python
a = await rx.ask_all(ticket, category=category, severity=severity,
                     repro=has_repro, refund=refund_requested)
if a["category"] == "bug_report" and a["severity"].value > 1.5 and a["repro"].p > 0.6:
    escalate(ticket_id)
```

**Confidence gating.** Thresholds are per-action, not per-application, because the same
system should gate a read-only lookup and an irreversible write differently:

```python
from reflex_jev import Gate, READ_ONLY, DESTRUCTIVE

match action.gate(DESTRUCTIVE):
    case Gate.ACT:      execute(account_id)
    case Gate.CONFIRM:  ask_user_to_confirm(account_id)
    case Gate.ESCALATE: route_to_human(account_id)
```

**Honest confidence for nouls.** Jev returns *no* confidence for a noul — only a
probability. `NoulDecision.confidence` is derived by this library as `|p − 0.5| × 2` and
documented as such, rather than being passed off as the model's own calibrated signal.

**A linter for the documented jagged edges.** Jev's own model card lists where it breaks:
it cannot count or do arithmetic, reads dates as text rather than ordered quantities,
does poorly on hex/RGB/low-level encodings, degrades on double negatives and multi-hop
reasoning, and does not generate text at all. `reflex-jev` checks question wording against
those at declaration time:

```python
>>> noul("How many invoices are unpaid and which date comes first").lint()
[Lint(code='counting',        severity='error', ...),
 Lint(code='date-comparison', severity='error', ...)]
```

Policy is `Reflex(lint="warn")` by default; `"raise"` fails the build on errors, `"off"`
disables it. Each distinct question is linted once, at first use.

**Token budgeting.** Requests are packed against both documented limits — 64k for state
plus all questions, 32k for state plus the single longest question — and split across
concurrent requests when they overflow, rather than failing at the API.

**An injection guard for untrusted state.** Jev's model card is explicit that state is data
and is *not* treated as hostile: text engineered to argue for its own classification can
move answers. Wrap end-user content and it gets fenced in the serialized state, and a
detection question is issued against it:

```python
from reflex_jev import untrusted

view = rx.view({"ticket": untrusted(user_message), "policy": refund_policy})
answers = await view.all(department, frustration)
view.guard.p   # 0-1 probability that the wrapped content is an injection attempt
```

The guard runs against the *extracted* untrusted fragments, not the whole state. That
detail matters more than it sounds. Asking Jev to locate "the content marked untrusted"
inside a nested object is a multi-hop question over irrelevant context — the two things
its model card says degrade accuracy most — and measured against `jev-1.13.0` it inflated
false positives badly:

| guard runs against | benign message | injection attempt |
|---|---|---|
| full nested state | 0.64 | 0.98 |
| extracted fragments | **0.08** | **0.98** |

Same benign text either way. Checking the fragments alone leaves the 0.7 default threshold
a wide margin instead of 0.06. It costs a second request, but one issued in the same tick
so it runs concurrently, and carrying only the untrusted text, so it is very cheap. The
guard only runs at all when you actually wrap something.

Above the threshold it warns; `Reflex(guard="raise")` raises `InjectionSuspected`, and
`guard=False` disables it.

## Install

```bash
pip install reflex-jev
export TYPESAFE_API_KEY="sk-..."
```

Requires Python 3.10+. Pin the model if you tune thresholds against it — `jev-latest`
changes without notice:

```python
rx = Reflex(model="jev-1.13.0")
```

## Batching across awaits

Automatic batching spans one event-loop tick. When the questions you want batched are
separated by `await` points, use an explicit scope:

```python
async with rx.batch(state) as v:
    dept = v.ask(department)
    rows = await load_from_db()          # an await in the middle
    anger = v.ask(frustration)
# both flush together here, as one request
```

## What this is not

Jev does not generate text, and neither does this library. It classifies, scores, routes
and verifies. Extract candidates with code or an LLM; let Jev pick among them.

## License

MIT

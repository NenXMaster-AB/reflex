"""reflex_jev -- batching, caching and confidence-gated control flow for Jev.

Jev evaluates every question in a request in parallel against one shared state,
bills only input tokens, and gives output away free. The economical shape is
therefore "many questions, one state, one request" -- and the naive shape, a
semantic check sprinkled at each call site, pays for the same state over and
over. This library makes the economical shape the one you get by default.
"""

from .cache import Cache, LRUCache, NullCache
from .client import (
    INJECTION_GUARD,
    Batch,
    Reflex,
    View,
)
from .decisions import (
    DESTRUCTIVE,
    READ_ONLY,
    STANDARD,
    ChoiceDecision,
    Decision,
    Gate,
    NoulDecision,
    Policy,
    ScoreDecision,
    composite,
)
from .errors import (
    AnswerMissing,
    BudgetError,
    InjectionSuspected,
    LintError,
    ReflexError,
)
from .questions import (
    ChoiceQuestion,
    Lint,
    NoulQuestion,
    Question,
    QuestionError,
    ScoreQuestion,
    choice,
    lint_all,
    noul,
    score,
)
from .scheduler import RequestStats
from .state import StateError, Untrusted, estimate_tokens, untrusted

__version__ = "0.1.0"

__all__ = [
    "Reflex",
    "View",
    "Batch",
    "noul",
    "choice",
    "score",
    "Question",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "Decision",
    "NoulDecision",
    "ChoiceDecision",
    "ScoreDecision",
    "composite",
    "Gate",
    "Policy",
    "READ_ONLY",
    "STANDARD",
    "DESTRUCTIVE",
    "untrusted",
    "Untrusted",
    "estimate_tokens",
    "Cache",
    "LRUCache",
    "NullCache",
    "RequestStats",
    "Lint",
    "lint_all",
    "INJECTION_GUARD",
    "ReflexError",
    "BudgetError",
    "LintError",
    "QuestionError",
    "StateError",
    "AnswerMissing",
    "InjectionSuspected",
    "__version__",
]

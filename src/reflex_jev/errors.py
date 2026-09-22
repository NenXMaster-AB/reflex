"""Exceptions raised by reflex_jev itself, distinct from SDK transport errors."""

from __future__ import annotations


class ReflexError(Exception):
    """Base class for reflex_jev errors."""


class BudgetError(ReflexError):
    """Raised when state cannot fit inside Jev's documented context limits."""


class LintError(ReflexError):
    """Raised when a question trips a lint rule under ``lint='raise'``."""

    def __init__(self, label: str, findings: list) -> None:
        detail = "; ".join(str(f) for f in findings)
        super().__init__(f"question {label!r} failed lint: {detail}")
        self.label = label
        self.findings = findings


class AnswerMissing(ReflexError):
    """Raised when Jev's response omits a question that was asked."""


class InjectionSuspected(ReflexError):
    """Raised when the injection guard fires on untrusted state."""

    def __init__(self, probability: float, threshold: float) -> None:
        super().__init__(
            f"untrusted state appears to contain injected instructions "
            f"(p={probability:.2f} >= {threshold:.2f})"
        )
        self.probability = probability
        self.threshold = threshold

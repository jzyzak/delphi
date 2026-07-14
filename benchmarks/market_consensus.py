"""Market-consensus adapter (C7.4).

Maps prediction-market records into pinned benchmark shapes and exposes the
market price *as of the pin* as a mandatory consensus baseline (CLAUDE.md §2.3:
additive information vs consensus is the honest, publishable claim). The baseline
hook builds an :class:`~evaluation.baselines.Baseline` scored through the same
harness path as the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchmarks.base import BenchmarkAdapter, BenchmarkQuestion, BenchmarkResolution, parse_dt
from evaluation.baselines import MARKET_CONSENSUS, Baseline, market_consensus

__all__ = ["MarketConsensusAdapter", "consensus_baseline"]

_SOURCE = "market_consensus"
_PRICE_KEY = "market_price"


class MarketConsensusAdapter:
    """A prediction-market question set built from fetched records."""

    def __init__(
        self,
        questions: Sequence[BenchmarkQuestion],
        resolutions: Sequence[BenchmarkResolution],
    ) -> None:
        self._questions = tuple(questions)
        self._resolutions = tuple(resolutions)

    @property
    def name(self) -> str:
        return _SOURCE

    def questions(self) -> Sequence[BenchmarkQuestion]:
        return self._questions

    def resolutions(self) -> Sequence[BenchmarkResolution]:
        return self._resolutions

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]]) -> MarketConsensusAdapter:
        """Map raw market records into questions + resolutions.

        Each record needs ``id``, ``question`` (text), ``as_of``, and ``price``
        (the market's implied probability at the pin). ``resolved_value`` +
        ``resolved_at`` yield a resolution.
        """
        questions: list[BenchmarkQuestion] = []
        resolutions: list[BenchmarkResolution] = []
        for record in records:
            question = BenchmarkQuestion(
                source=_SOURCE,
                external_id=str(record["id"]),
                text=str(record["question"]),
                as_of=parse_dt(record["as_of"]),
                question_type="binary",
                domain=str(record.get("domain", "markets")),
                resolution_criteria=str(record.get("resolution_criteria", "")),
                close_time=parse_dt(record["close_time"]) if record.get("close_time") else None,
                metadata={"benchmark": _SOURCE, _PRICE_KEY: float(record["price"])},
            )
            questions.append(question)
            if record.get("resolved_value") is not None and record.get("resolved_at"):
                resolutions.append(
                    BenchmarkResolution(
                        question_id=question.question_id,
                        resolved_value=float(record["resolved_value"]),
                        resolved_at=parse_dt(record["resolved_at"]),
                        source=_SOURCE,
                    )
                )
        return cls(questions, resolutions)


def consensus_baseline(adapter: BenchmarkAdapter, *, price_key: str = _PRICE_KEY) -> Baseline:
    """Build the market-consensus baseline from each question's pinned price."""
    predictions: dict[str, float] = {}
    for question in adapter.questions():
        price = question.metadata.get(price_key)
        if price is not None:
            predictions[question.question_id] = market_consensus(float(price))
    return Baseline(name=MARKET_CONSENSUS, predictions=predictions)

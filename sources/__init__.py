"""As-of evidence providers (implement the core search Protocol).

App layer (not shared ``core/``): DELPHI-specific retrieval that turns a query +
as-of ceiling into leakage-safe :class:`~core.forecast.search.Evidence`, snapshot
it for reproducibility, and expose it as an ``AsOfSearcher``.
"""

from __future__ import annotations

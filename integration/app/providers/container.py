"""Request-scoped provider collection used by the integration pipeline."""

from dataclasses import dataclass, field


@dataclass
class Providers:
    """Hold the concrete providers selected for one request.

    Provider objects intentionally use duck typing. JSON contracts and
    subprocess tests already validate the integration boundary, so an ABC
    hierarchy only duplicated those checks and encouraged global mutation.
    """

    paper_source: object | None = None
    parser: object | None = None
    recommender: object | None = None
    synthesizer: object | None = None
    source_labels: dict[str, str] = field(default_factory=dict)

    def add(self, role: str, provider: object, source: str) -> "Providers":
        if role not in {"paper_source", "parser", "recommender", "synthesizer"}:
            raise ValueError(f"Unknown provider role: {role}")
        setattr(self, role, provider)
        self.source_labels[role] = source
        return self

    def require(self, role: str):
        provider = getattr(self, role, None)
        if provider is None:
            raise RuntimeError(
                f"No provider configured for {role!r}. "
                "Production requests must configure providers explicitly."
            )
        return provider

    def source(self, role: str) -> str:
        if role not in self.source_labels:
            raise RuntimeError(f"No provider configured for {role!r}")
        return self.source_labels[role]

    def sources(self) -> dict[str, str]:
        return dict(self.source_labels)


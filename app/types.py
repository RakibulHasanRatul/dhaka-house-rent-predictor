from dataclasses import dataclass, field


@dataclass
class TrainingVector:
    feature_vectors: list[list[float]] = field(
        default_factory=list[list[float]]
    )
    labels: list[float] = field(default_factory=list[float])

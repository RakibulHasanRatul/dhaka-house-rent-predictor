import json
import os

from app.types import TrainingVector
from config import DATA_PROCESSED_DIR, WEIGHTS_JSON_DIR

from .linear_regression import get_weight_vector

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def train_data(dataset: dict[str, TrainingVector]):
    weights: dict[str, list[list[float]]] = {}

    locations = dataset.keys()

    for location in locations:
        weights[location] = get_weight_vector(
            feature_matrix=dataset[location].feature_vectors,
            labels=dataset[location].labels,
        )

    with open(WEIGHTS_JSON_DIR, "w") as file:
        json.dump(weights, file)

    return locations

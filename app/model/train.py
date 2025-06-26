import json
import os
from typing import Callable

from app.types import TrainingVector
from config import DATA_PROCESSED_DIR, WEIGHTS_JSON_DIR

from .linear_regression import model_train

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def train_all_dataset(
    dataset: dict[str, TrainingVector],
    train_fn: Callable[
        [list[list[float]], list[list[float]], float], list[list[float]]
    ] = model_train,
):
    weights: dict[str, list[list[float]]] = {}

    locations = dataset.keys()

    for location in locations:
        weights[location] = train_fn(
            dataset[location].feature_vectors,
            [[y] for y in dataset[location].labels],
            1e-13,
        )

    with open(WEIGHTS_JSON_DIR, "w") as file:
        json.dump(weights, file)

    return locations

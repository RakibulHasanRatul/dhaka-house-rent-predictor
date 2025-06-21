import json
import os

from app.types import TrainingVector
from config import DATA_PROCESSED_DIR, WEIGHTS_JSON_DIR

from .linear_regression import model_train

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def train_all_dataset(dataset: dict[str, TrainingVector]):
    weights: dict[str, list[list[float]]] = {}

    locations = dataset.keys()

    for location in locations:
        weights[location] = model_train(
            x_vector=dataset[location].feature_vectors,
            y_vector=[[y] for y in dataset[location].labels],
        )

    with open(WEIGHTS_JSON_DIR, "w") as file:
        json.dump(weights, file)

    return locations

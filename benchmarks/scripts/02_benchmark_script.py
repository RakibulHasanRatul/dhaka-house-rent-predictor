import os
import sys

sys.path.insert(0, os.getcwd())

import json
from collections import defaultdict
from typing import Any

from __common import run_benchmark

from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.helper import construct_features_list
from app.types import TrainingVector
from config import (
    DATA_PROCESSED_DIR,
    FORMATTED_CSV_GIST_URL,
    LOCATION_JSON_DIR,
    TYPES_JSON_DIR,
)


def modified_construct_location_from_area(addr: str) -> str:
    parts = [part.strip() for part in addr.split(",")]
    if len(parts) < 2:
        return addr.title()

    return f"{parts[-2]}, {parts[-1]}".title()


def modified_preprocess_loaded_data(
    data: dict[str, list[Any]],
) -> dict[str, TrainingVector]:
    house_types = sorted({str(cell).strip() for cell in data["type"]})
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(TYPES_JSON_DIR, "w") as f:
        json.dump({"types": list(house_types)}, f)

    data["type_num"] = [house_types.index(str(cell).strip()) for cell in data["type"]]
    del data["type"]

    data["location"] = [
        modified_construct_location_from_area(str(cell)) for cell in data["address"]
    ]
    del data["address"]

    locations = sorted({str(cell) for cell in data["location"]})
    with open(LOCATION_JSON_DIR, "w") as f:
        json.dump({"locations": list(locations)}, f)

    rents: dict[str, list[float]] = defaultdict(list[float])
    features: dict[str, list[list[float]]] = defaultdict(list[list[float]])

    for i, location in enumerate(data["location"]):
        feature_set = [1.0] + construct_features_list(
            beds=float(data["beds"][i]),
            bath=float(data["bath"][i]),
            area=float(data["area"][i]),
            type_num=float(data["type_num"][i]),
            year=float(data["year"][i]),
        )
        features[location].append(feature_set)
        rents[location].append(float(data["rent"][i]))

    training_dataset: dict[str, TrainingVector] = {}
    for location in locations:
        training_dataset[location] = TrainingVector(
            feature_vectors=features[location], labels=rents[location]
        )

    return training_dataset


if __name__ == "__main__":
    run_benchmark(
        preprocessed_data=modified_preprocess_loaded_data(
            load_csv_data(download_csv_from_gist(FORMATTED_CSV_GIST_URL))
        )
    )

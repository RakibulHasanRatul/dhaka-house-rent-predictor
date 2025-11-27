from collections import defaultdict
from typing import Any
from app.helper import construct_features_list
from app.types import TrainingVector


def modified_construct_location_from_area(addr: str) -> str:
    parts = [part.strip() for part in addr.split(",")]
    if len(parts) < 2:
        return addr.title()

    return f"{parts[-2]}, {parts[-1]}".title()


def modified_preprocess_loaded_data(
    data: dict[str, list[Any]],
) -> dict[str, TrainingVector]:
    house_types = sorted({str(cell).strip() for cell in data["type"]})

    data["type_num"] = [house_types.index(str(cell).strip()) for cell in data["type"]]
    del data["type"]

    data["location"] = [
        modified_construct_location_from_area(str(cell)) for cell in data["address"]
    ]
    del data["address"]

    locations = sorted({str(cell) for cell in data["location"]})
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

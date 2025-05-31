import json
import os
from collections import defaultdict
from typing import Any

from config import DATA_PROCESSED_DIR, LOCATION_JSON_DIR, TYPES_JSON_DIR

from ...helper import construct_features_list
from ...types import TrainingVector
from .load import load_csv_data


def text_to_num(text: str) -> float:
    mapping = {"thousand": 1_000, "lakh": 100_000}
    parts = text.split()
    number = float(parts[0])
    suffix = parts[1].lower() if len(parts) > 1 else ""
    multiplier = mapping.get(suffix, 1)
    return float(number * multiplier)


def construct_location_from_area(addr: str) -> str:
    parts = [part.strip() for part in addr.split(",")]
    if len(parts) < 3:
        if len(parts) == 1:
            return addr.title()
        return f"{parts[-2]}, {parts[-1]}".title()

    level3 = parts[-3]
    level2 = parts[-2]
    level1 = parts[-1]

    return f"{level3}, {level2}, {level1}".title()


def preprocess_loaded_data(data: dict[str, list[Any]]) -> dict[str, TrainingVector]:
    house_types = sorted({str(cell).strip() for cell in data["type"]})
    # dump to json file, it will help to construct the ui later on
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(TYPES_JSON_DIR, "w") as f:
        json.dump({"types": list(house_types)}, f)

    # for prediction the types need to be as a numeric representation,
    # this type_num column will do so.
    data["type_num"] = [house_types.index(str(cell).strip()) for cell in data["type"]]
    del data["type"]

    # I shall be working last 3 levels of address only,
    # so I'll create a new column named 'location' to do so
    data["location"] = [
        construct_location_from_area(str(cell)) for cell in data["address"]
    ]
    del data["address"]

    # I'll save the unique locations to a json file
    # so that it's easy to construct the ui later on
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


def load_and_refit_kaggle_csv(file_path: str) -> dict[str, list[Any]]:
    loaded_data = load_csv_data(file_path)

    # firstly I'll delete unnecessary data!
    for column in ["title", "purpose", "flooPlan", "url"]:
        del loaded_data[column]

    # I saw data like "March 15, 2022" in the lastUpdated column.
    # I can extract the year value from there.
    loaded_data["year"] = [
        int(cell.split(",")[-1].strip()) for cell in loaded_data["lastUpdated"]
    ]
    del loaded_data["lastUpdated"]

    # the file i picked from kaggle has some weird data. in beds column
    # it has 1 bed, 2 beds etc. type of data. same goes for bath.
    # even in the area column, number with sq ft (the unit) is added,
    # i need to fit those data.
    for column in ["area", "beds", "bath"]:
        for i, cell in enumerate(loaded_data[column]):
            loaded_data[column][i] = float(str(cell).split(" ")[0].replace(",", ""))

    # rename the adress column to address: spelling mistake!
    loaded_data["address"] = loaded_data.pop("adress")

    # I'll work only with Dhaka. So I'll delete data if not of Dhaka.
    columns = list(loaded_data.keys())
    for i in reversed(range(len(loaded_data["address"]))):
        if "dhaka" not in str(loaded_data["address"][i]).lower():
            for column in columns:
                loaded_data[column].pop(i)

    # rename price to rent! It makes sense better! although it was unnecessary, whatever...
    loaded_data["rent"] = [text_to_num(str(cell)) for cell in loaded_data["price"]]
    del loaded_data["price"]

    return loaded_data

import csv
from collections import defaultdict
from typing import Any


def load_csv_data(file_path: str) -> dict[str, list[Any]]:
    data: dict[str, list[Any]] = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    print(row)
                data[k].append(v)
    return dict(data)

import os
import sys

sys.path.insert(0, os.getcwd())

from py_impl import train as model_train

from app.types import TrainingVector
from benchmarks.scripts.metrics_func import r_squared


def generate_scoreboard(preprocessed_data: dict[str, TrainingVector]):
    scoreboard: dict[str, float] = {}

    for location, data in preprocessed_data.items():
        x_total = data.feature_vectors
        y_total = data.labels
        total = len(x_total)

        if total >= 5:
            slice = total // 5

            x_test = x_total[:slice]
            y_test = y_total[:slice]

            x_train = x_total[slice:]
            y_train = y_total[slice:]

            weights = model_train(x_train, [[y] for y in y_train])

            y_pred: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                y_pred.append(pred)

            scoreboard[location] = round(r_squared(y_pred, y_test) * 10, 2)

        else:
            scoreboard[location] = -1e13
            # ensuring the least value possible!

    print("| Location | Score |")
    print("| --- | --- |")

    locations = list(scoreboard.keys())

    locations = sorted(locations, key=lambda x: scoreboard[x], reverse=True)

    for location in locations:
        print(
            f"|{location}|{scoreboard[location] if scoreboard[location] != -1e13 else 'N/A'}|"
        )

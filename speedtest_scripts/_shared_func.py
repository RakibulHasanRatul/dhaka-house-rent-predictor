import os
import sys

sys.path.insert(0, os.getcwd())

from sklearn.linear_model import LinearRegression  # type:ignore
import timeit
from app.model.linear_regression import get_weight_vector
from benchmarks.scripts.__common import r_squared, mse, mae
from app.types import TrainingVector


def run_speedtest(preprocessed_data: dict[str, TrainingVector]):
    locations = list(preprocessed_data.keys())

    work_on_location = [
        location
        for location in locations
        if len(preprocessed_data[location].feature_vectors) >= 10
    ]

    print("Working on {} locations.".format(len(work_on_location)))
    print(
        "Skipped {} locations due to insufficient data.".format(
            len(locations) - len(work_on_location)
        )
    )

    def run_scratch_built_model():
        for location in work_on_location:
            data = preprocessed_data[location]

            x_total = data.feature_vectors
            y_total = data.labels
            total = len(x_total)

            k = 5
            fold_size = total // k

            for fold in range(k):
                start = fold * fold_size
                end = start + fold_size

                x_test = x_total[start:end]
                y_test = y_total[start:end]

                x_train = x_total[:start] + x_total[end:]
                y_train = y_total[:start] + y_total[end:]

                weights = get_weight_vector(x_train, y_train)

                y_pred: list[float] = []
                for x in x_test:
                    pred = sum(w[0] * xi for w, xi in zip(weights, x))
                    y_pred.append(pred)

                r_squared(y_pred, y_test)
                mse(y_pred, y_test)
                mae(y_pred, y_test)

    def run_sklearn_model():
        for location in work_on_location:
            data = preprocessed_data[location]

            x_total = data.feature_vectors
            y_total = data.labels
            total = len(x_total)

            k = 5
            fold_size = total // k

            for fold in range(k):
                start = fold * fold_size
                end = start + fold_size

                x_test = x_total[start:end]
                y_test = y_total[start:end]

                x_train = x_total[:start] + x_total[end:]
                y_train = y_total[:start] + y_total[end:]

                model = LinearRegression()
                model.fit(x_train, y_train)  # type:ignore

                y_pred: list[float] = model.predict(x_test).tolist()  # type:ignore

                r_squared(y_pred, y_test)  # type:ignore
                mse(y_pred, y_test)  # type:ignore
                mae(y_pred, y_test)  # type:ignore

    print("Running scratch-built model...")
    scratch_time = timeit.timeit(run_scratch_built_model, number=1)
    print(f"Scratch-built model took {scratch_time:.4f} seconds.")
    print("Running sklearn model...")
    sklearn_time = timeit.timeit(run_sklearn_model, number=1)
    print(f"Sklearn model took {sklearn_time:.4f} seconds.")

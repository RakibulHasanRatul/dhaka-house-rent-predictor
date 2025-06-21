import os
import sys
import timeit

sys.path.insert(0, os.getcwd())

from sklearn.linear_model import LinearRegression  # type:ignore

from _shared_func import load_all_datasets
from app.model.linear_regression import model_train
from performance_metrics_functions import mae, mse, r_squared


def run_speedtest_ensuring_larger_datasets():
    x_total, y_total = load_all_datasets()
    dataset_len = len(x_total)

    print("Working on {} dataset.".format(dataset_len))

    def run_scratch_built_model():
        k = 5
        fold_size = dataset_len // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size

            x_test = x_total[start:end]
            y_test = y_total[start:end]

            x_train = x_total[:start] + x_total[end:]
            y_train = y_total[:start] + y_total[end:]

            weights = model_train(x_train, [[y] for y in y_train])

            y_pred: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                y_pred.append(pred)

            r_squared(y_pred, y_test)
            mse(y_pred, y_test)
            mae(y_pred, y_test)

    def run_sklearn_model():
        k = 5
        fold_size = dataset_len // k

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


if __name__ == "__main__":
    run_speedtest_ensuring_larger_datasets()

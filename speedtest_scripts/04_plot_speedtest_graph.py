import os
import sys

sys.path.insert(0, os.getcwd())

from sklearn.linear_model import LinearRegression  # type:ignore
import timeit
from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.model.linear_regression import model_train
from config import FORMATTED_CSV_GIST_URL
from benchmarks.scripts.__common import (
    r_squared,
    mse,
    mae,
    modified_preprocess_loaded_data,
)
import matplotlib.pyplot as plt

NUMBER_OF_FOLDS = int(os.environ.get("CV_N", 5))
DATASET_LEN_INCREASE = NUMBER_OF_FOLDS  # of course can be changed!
# making these global const so that it can be easy to tweak!


def run_speedtest_and_plot_graph():
    preprocessed_data = modified_preprocess_loaded_data(
        load_csv_data(download_csv_from_gist(FORMATTED_CSV_GIST_URL))
    )

    locations = list(preprocessed_data.keys())

    x_total: list[list[float]] = []
    y_total: list[float] = []

    for location in locations:
        x_total = x_total + preprocessed_data[location].feature_vectors
        y_total = y_total + preprocessed_data[location].labels

    dataset_len_total = len(x_total)

    used_dataset_lengths: list[int] = []
    scratch_built_times: list[float] = []
    sklearn_times: list[float] = []

    def run_scratch_built_model(
        dataset_len: int, feature_vector: list[list[float]], label_vector: list[float]
    ):
        fold_size = dataset_len // NUMBER_OF_FOLDS

        for fold in range(NUMBER_OF_FOLDS):
            start = fold * fold_size
            end = start + fold_size

            x_test = feature_vector[start:end]
            y_test = label_vector[start:end]

            x_train = feature_vector[:start] + feature_vector[end:]
            y_train = label_vector[:start] + label_vector[end:]

            weights = model_train(x_train, y_train)

            y_pred: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                y_pred.append(pred)

            r_squared(y_pred, y_test)
            mse(y_pred, y_test)
            mae(y_pred, y_test)

    def run_sklearn_model(
        dataset_len: int, feature_vector: list[list[float]], label_vector: list[float]
    ):
        fold_size = dataset_len // NUMBER_OF_FOLDS

        for fold in range(NUMBER_OF_FOLDS):
            start = fold * fold_size
            end = start + fold_size

            x_test = feature_vector[start:end]
            y_test = label_vector[start:end]

            x_train = feature_vector[:start] + feature_vector[end:]
            y_train = label_vector[:start] + label_vector[end:]

            model = LinearRegression()
            model.fit(x_train, y_train)  # type:ignore

            y_pred: list[float] = model.predict(x_test).tolist()  # type:ignore

            r_squared(y_pred, y_test)  # type:ignore
            mse(y_pred, y_test)  # type:ignore
            mae(y_pred, y_test)  # type:ignore

    dataset_len = NUMBER_OF_FOLDS
    while True:
        # yea, I could use conditions like dataset_len <= dataset_len_total,
        # but I want to use the full datasets
        print("Working on {} dataset.".format(dataset_len))
        print("Running scratch-built model with {} dataset...".format(dataset_len))
        scratch_time = timeit.timeit(
            lambda: run_scratch_built_model(
                dataset_len, x_total[:dataset_len], y_total[:dataset_len]
            ),
            number=1,
        )
        print(f"Scratch-built model took {scratch_time:.4f} seconds.")
        print("Running sklearn model with {} dataset...".format(dataset_len))
        sklearn_time = timeit.timeit(
            lambda: run_sklearn_model(dataset_len, x_total[:dataset_len], y_total[:dataset_len]),
            number=1,
        )
        print(f"Sklearn model took {sklearn_time:.4f} seconds.")

        used_dataset_lengths.append(dataset_len)
        scratch_built_times.append(scratch_time)
        sklearn_times.append(sklearn_time)

        if dataset_len == dataset_len_total:
            break

        dataset_len += DATASET_LEN_INCREASE

        if dataset_len > dataset_len_total:
            dataset_len = dataset_len_total

    plt.figure(figsize=(16, 9))  # type:ignore
    # 16:9 looks fine to me!
    plt.plot(  # type:ignore
        used_dataset_lengths,
        scratch_built_times,
        label="Scratch-built model",
    )
    plt.plot(  # type:ignore
        used_dataset_lengths,
        sklearn_times,
        label="scikit-learn model",
    )
    plt.xlabel("Dataset Length")  # type:ignore
    plt.ylabel("Time (seconds)")  # type:ignore
    plt.title("Speed Comparison of Linear Regression Models")  # type:ignore
    plt.grid(True)  # type:ignore
    plt.legend()  # type:ignore
    plt.tight_layout()  # type:ignore
    os.makedirs(os.path.join(os.getcwd(), "images/graphs/speed_comparisons"), exist_ok=True)
    plt.savefig(  # type:ignore
        f"images/graphs/speed_comparisons/{NUMBER_OF_FOLDS}_fold_speed_comparison.png",
        dpi=300,
    )


if __name__ == "__main__":
    run_speedtest_and_plot_graph()

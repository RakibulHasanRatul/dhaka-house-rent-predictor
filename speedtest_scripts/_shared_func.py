import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())

import timeit
from typing import Any, Callable

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # type:ignore

from app.handler.data.download import download_csv_from_gist
from app.helper import construct_features_list
from app.model.linear_regression import model_train
from app.types import TrainingVector
from config import FORMATTED_CSV_GIST_URL
from performance_metrics_functions import mae, mse, r_squared


def run_scratch_model(
    dataset_len: int,
    feature_vector: list[list[float]],
    label_vector: list[float],
    n_folds: int,
    train_func: Callable[..., list[list[float]]],
    predict_func: Callable[[list[float], list[list[float]]], float],
    reg: float,
):
    fold_size = dataset_len // n_folds

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size

        x_test = feature_vector[start:end]
        y_test = label_vector[start:end]

        x_train = feature_vector[:start] + feature_vector[end:]
        y_train = label_vector[:start] + label_vector[end:]

        weights = train_func(x_train, [[y] for y in y_train], reg)

        y_pred: list[float] = []
        for x in x_test:
            pred = predict_func(x[1:], weights)
            y_pred.append(pred)

        r_squared(y_pred, y_test)
        mse(y_pred, y_test)
        mae(y_pred, y_test)


def run_sklearn_model(
    dataset_len: int,
    feature_vector: list[list[float]],
    label_vector: list[float],
    n_folds: int,
):
    fold_size = dataset_len // n_folds

    for fold in range(n_folds):
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

    def run_scratch_lr_model():
        for location in work_on_location:
            data = preprocessed_data[location]

            x_total = data.feature_vectors
            y_total = data.labels
            total = len(x_total)

            run_scratch_model(
                total,
                x_total,
                y_total,
                5,
                model_train,
                lambda x, w: sum(w[0] * xi for w, xi in zip(w, x)),
                1e-12,
            )

    def run_sklearn_lr_model():
        for location in work_on_location:
            data = preprocessed_data[location]

            x_total = data.feature_vectors
            y_total = data.labels
            total = len(x_total)

            run_sklearn_model(total, x_total, y_total, 5)

    print("Running scratch-built model...")
    scratch_time = timeit.timeit(run_scratch_lr_model, number=1)
    print(f"Scratch-built model took {scratch_time:.4f} seconds.")
    print("Running sklearn model...")
    sklearn_time = timeit.timeit(lambda: run_sklearn_lr_model(), number=1)
    print(f"Sklearn model took {sklearn_time:.4f} seconds.")


def load_all_datasets() -> tuple[list[list[float]], list[float]]:
    file_path = download_csv_from_gist(FORMATTED_CSV_GIST_URL)
    data: dict[str, list[Any]] = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as file:
        import csv

        reader = csv.DictReader(file)
        for row in reader:
            for k, v in row.items():
                data[k].append(v)

    house_types = sorted({str(cell).strip() for cell in data["type"]})
    data["type_num"] = [house_types.index(str(cell).strip()) for cell in data["type"]]
    del data["type"]
    del data["address"]

    features: list[list[float]] = []
    rents: list[float] = []

    for i in range(len(data["type_num"])):
        features.append(
            [1.0]
            + construct_features_list(
                beds=float(data["beds"][i]),
                bath=float(data["bath"][i]),
                area=float(data["area"][i]),
                type_num=float(data["type_num"][i]),
                year=float(data["year"][i]),
            )
        )
        rents.append(float(data["rent"][i]))

    return (features, rents)


def run_speedtest_and_plot_graph(
    img_path: str,
    x_total: list[list[float]],
    y_total: list[float],
    n_folds: int,
    dataset_len_increment: int,
    scratch_train_func: Callable[..., list[list[float]]],
    scratch_predict_func: Callable[[list[float], list[list[float]]], float],
    reg_scratch: float = 1e-13,
    scratch_graph_label: str = "Scratch-built model",
):
    dataset_len_total = len(x_total)

    used_dataset_lengths: list[int] = []
    scratch_built_times: list[float] = []
    sklearn_times: list[float] = []

    dataset_len = n_folds
    while True:
        # yea, I could use conditions like dataset_len <= dataset_len_total,
        # but I want to use the full datasets
        print("Working on {} dataset.".format(dataset_len))
        print("Running scratch model with {} dataset...".format(dataset_len))
        scratch_time = timeit.timeit(
            lambda: run_scratch_model(
                dataset_len,
                x_total[:dataset_len],
                y_total[:dataset_len],
                n_folds,
                scratch_train_func,
                scratch_predict_func,
                reg_scratch,
            ),
            number=1,
        )
        print(f"Scratch model took {scratch_time:.4f} seconds.")
        print("Running sklearn model with {} dataset...".format(dataset_len))
        sklearn_time = timeit.timeit(
            lambda: run_sklearn_model(
                dataset_len, x_total[:dataset_len], y_total[:dataset_len], n_folds
            ),
            number=1,
        )
        print(f"Sklearn model took {sklearn_time:.4f} seconds.")

        used_dataset_lengths.append(dataset_len)
        scratch_built_times.append(scratch_time)
        sklearn_times.append(sklearn_time)

        if dataset_len == dataset_len_total:
            break

        dataset_len += dataset_len_increment

        if dataset_len > dataset_len_total:
            dataset_len = dataset_len_total

    fig, ax = plt.subplots(figsize=(16, 9))  # type:ignore
    # 16:9 looks fine to me!
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.plot(  # type:ignore
        used_dataset_lengths,
        sklearn_times,
        label="Scikit-Learn LinearRegression model",
        color="green",
    )
    ax.plot(  # type:ignore
        used_dataset_lengths,
        scratch_built_times,
        label=f"{scratch_graph_label}",
        color="red",
    )

    ax.text(  # type:ignore
        0.05,
        0.95,
        f"Number of Samples : {dataset_len_total}\n"
        f"Number of Features : {len(x_total[0])}\n"
        f"Number of CV Folds : {n_folds}\n",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=1),
        fontsize=12,
    )
    ax.set_xlabel("Dataset Length", labelpad=10, fontsize=12)  # type:ignore
    ax.set_ylabel("Time (seconds)", labelpad=10, fontsize=12)  # type:ignore
    ax.set_title("Speed Comparison of Linear Regression Models", pad=10, fontsize=16)  # type:ignore
    ax.grid(True)  # type:ignore
    ax.legend(fontsize=12)  # type:ignore
    plt.tight_layout(pad=5)  # type:ignore
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=300, bbox_inches="tight", pad_inches=0.5)  # type:ignore

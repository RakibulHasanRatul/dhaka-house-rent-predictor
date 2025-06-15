import json
import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression  # type:ignore

from app.helper import construct_features_list
from app.model.linear_regression import get_weight_vector
from app.types import TrainingVector
from config import DATA_PROCESSED_DIR, LOCATION_JSON_DIR, TYPES_JSON_DIR


def r_squared(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    # in the previous commits, I used the formula r_squared_value = 1 - (ss_res / ss_tot)
    # but to reduced the number of mathematical operations,
    # I used the formula r_squared_value = ss_reg / ss_tot
    # this formula is defined in wikipedia
    # you can learn more at:
    # https://en.wikipedia.org/wiki/Coefficient_of_determination#As_explained_variance

    y_mean = sum(y_original) / len(y_original)

    _q = sum((y_o - y_mean) ** 2 for y_o in y_original)
    if _q == 0:
        # All y values are (almost) the same, R squared value is undefined
        # treating as 0 for safe reporting
        return 0.0

    _p = sum((y_p - y_mean) ** 2 for y_p in y_predicted)

    return _p / _q


def mse(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    return sum(
        (float(y) - float(y_hat)) ** 2 for y, y_hat in zip(y_original, y_predicted)
    ) / len(y_original)


def mae(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    return sum(
        abs(float(y) - float(y_hat)) for y, y_hat in zip(y_original, y_predicted)
    ) / len(y_original)


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


def run_benchmark(preprocessed_data: dict[str, TrainingVector]):
    for location, data in preprocessed_data.items():
        print(f"\n\n🖈 {location}  ")
        x_total = data.feature_vectors
        y_total = data.labels
        total = len(x_total)

        if total < 10:
            print(f"⛌ Skipping {location} (not enough data points: {total})  ")
            continue
        else:
            print(f"Total data points: {total}")

        k = 5  # Number of folds for cross-validation
        fold_size = total // k

        print(
            "| Fold | R² (Scratch) | R² (Sklearn) | MSE (Scratch) | MSE (Sklearn) | MAE (Scratch) | MAE (Sklearn) |"
        )
        print("|---|---|---|---|---|---|---|")

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size

            x_test = x_total[start:end]
            y_test = y_total[start:end]

            x_train = x_total[:start] + x_total[end:]
            y_train = y_total[:start] + y_total[end:]

            weights = get_weight_vector(x_train, y_train)

            sklearn_model = LinearRegression()
            sklearn_model.fit(x_train, y_train)  # type:ignore

            y_pred: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                y_pred.append(pred)

            pred_sklearn = sklearn_model.predict(x_test)  # type:ignore

            print(
                "|"
                + "|".join(["{}"] * 7).format(
                    fold + 1,
                    round(r_squared(y_pred, y_test), 5),
                    round(r_squared((pred_sklearn), y_test), 5),  # type:ignore
                    round(mse(y_pred, y_test), 5),
                    round(mse(pred_sklearn, y_test), 5),  # type:ignore
                    round(mae(y_pred, y_test), 5),
                    round(mae(pred_sklearn, y_test), 5),  # type:ignore
                )
                + "|"
            )

        print("---")


def create_bar_plot(
    preprocessed_data: dict[str, TrainingVector], abs_graphs_dir: str, prefix: str = ""
):
    k = 5  # Number of folds for cross-validation
    scratch_r2_register: dict[int, dict[str, float]] = defaultdict(dict[str, float])
    sklearn_r2_register: dict[int, dict[str, float]] = defaultdict(dict[str, float])
    scratch_mse_register: dict[int, dict[str, float]] = defaultdict(dict[str, float])
    sklearn_mse_register: dict[int, dict[str, float]] = defaultdict(dict[str, float])
    scratch_mae_register: dict[int, dict[str, float]] = defaultdict(dict[str, float])
    sklearn_mae_register: dict[int, dict[str, float]] = defaultdict(dict[str, float])

    for location, data in preprocessed_data.items():
        x_total = data.feature_vectors
        y_total = data.labels
        total = len(x_total)

        if total < 10:
            continue

        fold_size = total // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size

            x_test = x_total[start:end]
            y_test = y_total[start:end]

            x_train = x_total[:start] + x_total[end:]
            y_train = y_total[:start] + y_total[end:]

            weights = get_weight_vector(x_train, y_train)

            sklearn_model = LinearRegression()
            sklearn_model.fit(x_train, y_train)  # type:ignore

            scratch_prediction: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                scratch_prediction.append(pred)

            sklearn_prediction = sklearn_model.predict(x_test)  # type:ignore

            scratch_r2_register[fold].update(
                {location: r_squared(scratch_prediction, y_test)}
            )
            sklearn_r2_register[fold].update(
                {location: r_squared(sklearn_prediction, y_test)}  # type:ignore
            )
            scratch_mse_register[fold].update(
                {location: mse(scratch_prediction, y_test)}
            )
            sklearn_mse_register[fold].update(
                {location: mse(sklearn_prediction, y_test)}  # type:ignore
            )
            scratch_mae_register[fold].update(
                {location: mae(scratch_prediction, y_test)}
            )
            sklearn_mae_register[fold].update(
                {location: mae(sklearn_prediction, y_test)}  # type:ignore
            )

    for fold in range(k):
        print("Creating Bar Plot for fold", fold + 1, "...")
        locations = list(scratch_r2_register[fold].keys())
        fig_width = 72
        fig_height = 48
        figure, axis = plt.subplots(  # type:ignore
            nrows=3,
            ncols=1,
            figsize=(fig_width, fig_height),
        )
        figure.suptitle(  # type:ignore
            f"Fold {fold + 1} Comparison: Scratch vs Scikit-learn", fontsize=32
        )

        metrics = [
            ("R Squared", scratch_r2_register, sklearn_r2_register),
            ("Mean Squared Error", scratch_mse_register, sklearn_mse_register),
            ("Mean Absolute Error", scratch_mae_register, sklearn_mae_register),
        ]

        for i, (label, scratch_data, sklearn_data) in enumerate(metrics):
            scratch_vals = [scratch_data[fold][loc] for loc in locations]
            sklearn_vals = [sklearn_data[fold][loc] for loc in locations]

            x = np.arange(len(locations))
            width = 0.35

            axis[i].bar(
                x - width / 2,
                scratch_vals,
                width,
                label="Scratch Implementation",
            )
            axis[i].bar(
                x + width / 2,
                sklearn_vals,
                width,
                label="Scikit-learn Implementation",
            )
            axis[i].set_ylabel(label)
            axis[i].set_title(f"{label} Comparison")
            axis[i].set_xticks(x)
            axis[i].set_xticklabels(
                [",".join(loc.split(",")[:-1]) for loc in locations],
                rotation=90,
                ha="right",
                fontsize=20,
            )
            axis[i].tick_params(
                axis="x",
                which="both",
                bottom=True,
                labelbottom=True,
                labelrotation=90,
                pad=20,
            )
            axis[i].legend()
            axis[i].grid(True, linestyle="--", alpha=0.5)

            if label == "R Squared":
                lower_bound = -1.5
                upper_bound = 1.0
            else:
                lower_bound = 0  # because mse and mae cannot be less than 0
                upper_bound = 1e7 if label == "Mean Squared Error" else 5e4

            padding = (upper_bound - lower_bound) * 0.1
            axis[i].set_ylim(lower_bound - padding, upper_bound + padding)

            for color, vals in [("red", scratch_vals), ("blue", sklearn_vals)]:
                for j, val in enumerate(vals):
                    if val < lower_bound or val > upper_bound:
                        y_pos = (
                            lower_bound + 0.05
                            if val < lower_bound
                            else upper_bound - 0.05
                        )
                        va = "bottom" if val < lower_bound else "top"

                        axis[i].annotate(
                            f"{val:.2f}",
                            xy=(x[j] - width / 2, y_pos),
                            textcoords="offset points",
                            xytext=(0, -5 if val < lower_bound else 5),
                            rotation=90,
                            ha="center",
                            va=va,
                            color=color,
                            fontsize=9,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="white",
                                ec="red",
                                lw=0.5,
                                alpha=0.8,
                            ),
                        )

        plt.tight_layout(rect=[0, 0.3, 1, 0.95], h_pad=5.0, w_pad=0.0)  # type:ignore
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.9,
            bottom=0.2,
            hspace=2,
        )
        os.makedirs(abs_graphs_dir, exist_ok=True)
        plt.savefig(  # type:ignore
            os.path.join(
                abs_graphs_dir,
                f"{prefix + ' ' if prefix else ''}fold_{fold + 1}_comparison.png",
            ),
            dpi=300,
        )
        plt.close(figure)
        print("Image saved...")

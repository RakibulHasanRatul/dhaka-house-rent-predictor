# type: ignore

from sklearn.linear_model import LinearRegression

from app.model.linear_regression import get_weight_vector
from app.types import TrainingVector


def r_squared(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    y_mean = sum(y_original) / len(y_original)

    ss_total = sum((y - y_mean) ** 2 for y in y_original)
    if ss_total == 0:
        # All y values are (almost) the same, R squared value is undefined
        # treating as 0 for safe reporting
        return 0.0

    ss_residual = sum(
        (y_o - float(y_p)) ** 2 for y_o, y_p in zip(y_original, y_predicted)
    )

    return 1 - (ss_residual / ss_total)


def mse(y_predicted: list[float], y_original: list[float]) -> float:
    return sum(
        (float(y) - float(y_hat)) ** 2 for y, y_hat in zip(y_original, y_predicted)
    ) / len(y_original)


def mae(y_predicted: list[float], y_original: list[float]) -> float:
    return sum(
        abs(float(y) - float(y_hat)) for y, y_hat in zip(y_original, y_predicted)
    ) / len(y_original)


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
            sklearn_model.fit(x_train, y_train)

            y_pred: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                y_pred.append(pred)

            pred_sklearn = sklearn_model.predict(x_test)

            print(
                "|"
                + "|".join(["{}"] * 7).format(
                    fold + 1,
                    round(r_squared(y_pred, y_test), 5),
                    round(r_squared((pred_sklearn), y_test), 5),
                    round(mse(y_pred, y_test), 5),
                    round(mse(pred_sklearn, y_test), 5),
                    round(mae(y_pred, y_test), 5),
                    round(mae(pred_sklearn, y_test), 5),
                )
                + "|"
            )

        print("---")

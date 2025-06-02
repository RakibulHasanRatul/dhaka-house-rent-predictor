import json

from config import WEIGHTS_JSON_DIR


def predict_rent(location: str, *features: float) -> float:
    with open(WEIGHTS_JSON_DIR, "r") as file:
        all_weights: dict[str, list[list[float]]] = json.load(file)

    desired_weights = all_weights[location]

    n = len(desired_weights)
    if len(features) != n - 1:
        raise ValueError(f"Expected {n} features, got {len(features)}")

    x = [1.0] + list(features)

    rent_prediction = 0.0
    for i in range(n):
        # the main formula is y = w^T.x (w is the weights vector, widely referred as theta)
        # but I don't waste computing resources to transpose it.
        # rather, using [i][0] is more efficient.
        rent_prediction += desired_weights[i][0] * x[i]

    return abs(rent_prediction)  # Ensure the prediction is non-negative

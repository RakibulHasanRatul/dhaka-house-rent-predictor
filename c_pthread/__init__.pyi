def train(
    feature_vectors: list[list[float]], label_vectors: list[list[float]], lambda_l2: float = 1e-13
) -> list[list[float]]:
    """
    Trains a linear regression model.

    Parameters
    ----------
    feature_vectors : list[list[float]]
        A list of feature vectors, where each feature vector is a list of floats. The feature vector is equivalent to [1, features[0], features[1], features[2], ..., features[n - 1]] where n is the number of features.
    label_vectors : list[list[float]]
        A list of labels, where each label is a float. it is equivalent to [[labels[0]], [labels[1]], ... [labels[n - 1]]] where n is the number of samples.
    lambda_l2 : float, optional
        The L2 regularization parameter, by default 1e-13.

    Returns
    -------
    list[list[float]]
        The learned weights as a list of lists of floats.
    """

def predict(features: list[float], weights: list[list[float]]) -> float:
    """
    Predicts the output using a linear regression model.

    Parameters
    ----------
    features : list[float]
        A list of input features, where each feature is a float. it is equivalent to [features[0], features[1], features[2], ..., features[n - 1]] where n is the number of features.
    weights : list[list[float]]
        The model weights, where each element is a list containing a single float. it is equivalent to [[weights[0]], [weights[1]], ... [weights[n - 1]]] where n is the number of features.

    Returns
    -------
    float
        The predicted output as a float value.
    """

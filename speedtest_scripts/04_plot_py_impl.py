import os
import sys

sys.path.insert(0, os.getcwd())


from _shared_func import load_all_datasets, run_speedtest_and_plot_graph
from app.model.linear_regression import model_train


NUMBER_OF_FOLDS = int(os.environ.get("NUMBER_OF_FOLDS", 5))
DATASET_LEN_INCREASE = int(os.environ.get("DATASET_LEN_INCREASE", NUMBER_OF_FOLDS * 3))
# of course can be changed!
# making these global const so that it can be easy to tweak!


if __name__ == "__main__":
    x_total, y_total = load_all_datasets()
    # not doing any feature incrementing or data incrementing, it's already taking much longer time!
    run_speedtest_and_plot_graph(
        f"images/graphs/speedtest_plots/py_impl_{NUMBER_OF_FOLDS}fold_{len(x_total)}d_{len(x_total[0])}f.png",
        x_total,
        y_total,
        NUMBER_OF_FOLDS,
        DATASET_LEN_INCREASE,
        model_train,
        lambda x_vector, weights: sum(w[0] * x for w, x in zip(weights, x_vector)),
    )

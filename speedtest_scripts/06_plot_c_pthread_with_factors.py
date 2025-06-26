import os
import sys

sys.path.insert(0, os.getcwd())

import random

from _shared_func import load_all_datasets, run_speedtest_and_plot_graph

from c_pthread import predict, train

NUMBER_OF_FOLDS = int(os.environ.get("NUMBER_OF_FOLDS", 5))
DATASET_LEN_INCREMENT = int(
    os.environ.get("DATASET_LEN_INCREMENT", NUMBER_OF_FOLDS * 3)
)
# of course can be changed!
FEATURE_GROWING_FACTOR = int(os.environ.get("FEATURE_GROWING_FACTOR", 1))
DATA_GROWING_FACTOR = int(os.environ.get("DATA_GROWING_FACTOR", 1))
# making these global const so that it can be easy to tweak!

if __name__ == "__main__":
    random.seed(2025)
    x_total, y_total = load_all_datasets()
    if FEATURE_GROWING_FACTOR > 1:
        x_total = [
            [
                x + i * random.uniform(-0.1, 0.1)
                for i in range(FEATURE_GROWING_FACTOR)
                for x in x_elem
            ]
            for x_elem in x_total
        ]
    # # Why? x_total=[x*5 for x in x_total] creates a singular matrix!
    if DATA_GROWING_FACTOR > 1:
        x_total = x_total * DATA_GROWING_FACTOR
        y_total = y_total * DATA_GROWING_FACTOR

    run_speedtest_and_plot_graph(
        f"images/graphs/speedtest_plots/c_pthread_{NUMBER_OF_FOLDS}fold_{len(x_total)}d_{len(x_total[0])}f.png",
        x_total,
        y_total,
        NUMBER_OF_FOLDS,
        DATASET_LEN_INCREMENT,
        train,
        predict,
        scratch_graph_label="Scratch-built [c_pthread] Model",
    )

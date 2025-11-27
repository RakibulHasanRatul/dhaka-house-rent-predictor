import os
import sys

sys.path.insert(0, os.getcwd())

from __common import create_bar_plot

from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import preprocess_loaded_data
from config import FORMATTED_CSV_GIST_URL

if __name__ == "__main__":
    create_bar_plot(
        preprocessed_data=preprocess_loaded_data(
            load_csv_data(download_csv_from_gist(FORMATTED_CSV_GIST_URL))
        ),
        abs_graphs_dir=os.path.join(
            os.getcwd(), "images/graphs/01_benchmark_results_bar_plot/"
        ),
        prefix="[01]",
    )

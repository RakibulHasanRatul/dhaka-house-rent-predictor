import os
import sys

sys.path.insert(0, os.getcwd())

from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import preprocess_loaded_data
from config import FORMATTED_CSV_GIST_URL

from _shared_func import run_speedtest


if __name__ == "__main__":
    run_speedtest(
        preprocessed_data=preprocess_loaded_data(
            load_csv_data(download_csv_from_gist(FORMATTED_CSV_GIST_URL))
        )
    )

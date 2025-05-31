from app.serve import serve_ui


from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import (
    preprocess_loaded_data,
    load_and_refit_kaggle_csv,
)
from app.model.train import train_data
from config import FORMATTED_CSV_GIST_URL, RAW_KAGGLE_CSV_GIST_URL


def load_data_and_train_model(from_raw_csv: bool = False, url: str = "") -> None:
    if not url:
        url = RAW_KAGGLE_CSV_GIST_URL if from_raw_csv else FORMATTED_CSV_GIST_URL

    file_path = download_csv_from_gist(url)

    if from_raw_csv:
        loaded_data = load_and_refit_kaggle_csv(file_path)

    else:
        loaded_data = load_csv_data(file_path)

    formatted_data = preprocess_loaded_data(loaded_data)

    train_data(formatted_data)


if __name__ == "__main__":
    # You can set it to True if wanna see how it performs with raw kaggle data I found.
    # Certainly, you can also set url parameter if you found a same-alike data and want to test.
    load_data_and_train_model(from_raw_csv=False)
    # after initialization, the program is gonna serve the ui!
    serve_ui()

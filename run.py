import os

from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import (
    load_and_refit_kaggle_csv,
    preprocess_loaded_data,
)
from app.model.train import train_all_dataset
from app.serve import serve_ui
from config import FORMATTED_CSV_GIST_URL, RAW_KAGGLE_CSV_GIST_URL


def load_data_and_train_model(
    from_raw_csv: bool = False, url: str = "", module: str = "py_impl"
) -> None:
    if not url:
        url = RAW_KAGGLE_CSV_GIST_URL if from_raw_csv else FORMATTED_CSV_GIST_URL

    file_path = download_csv_from_gist(url)

    if from_raw_csv:
        loaded_data = load_and_refit_kaggle_csv(file_path)

    else:
        loaded_data = load_csv_data(file_path)

    if module == "c_impl":
        from c_impl import train

        train_fn = train
    elif module == "c_pthread":
        from c_pthread import train

        train_fn = train
    else:
        from app.model.train import model_train

        train_fn = model_train

    formatted_data = preprocess_loaded_data(loaded_data)

    train_all_dataset(formatted_data, train_fn)


if __name__ == "__main__":
    # You can set it to True if wanna see how it performs with raw kaggle data I found.
    # Certainly, you can also set url parameter if you found a same-alike data and want to test.
    # Additionally, if you want to use the c implementation (pthead enabled),
    # then you can set use_c_pthread here.

    module_env = os.getenv("MODULE", "py_impl")

    load_data_and_train_model(from_raw_csv=False, module=module_env)
    # after initialization, the program is gonna serve the ui!
    serve_ui()

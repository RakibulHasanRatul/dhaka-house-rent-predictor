from enum import Enum
import sys

from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import (
    load_and_refit_kaggle_csv,
    preprocess_loaded_data,
)
from app.model.train import model_train, train_all_dataset
from app.serve import serve_ui
from config import FORMATTED_CSV_GIST_URL, RAW_KAGGLE_CSV_GIST_URL


class PackageList(Enum):
    py_impl = "py_impl"
    c_pthread = "c_pthread"
    c_impl = "c_impl"


def load_data_and_train_model(
    from_raw_csv: bool = False, url: str = "", module: PackageList = PackageList.py_impl
) -> None:
    if not url:
        url = RAW_KAGGLE_CSV_GIST_URL if from_raw_csv else FORMATTED_CSV_GIST_URL

    file_path = download_csv_from_gist(url)

    if from_raw_csv:
        loaded_data = load_and_refit_kaggle_csv(file_path)

    else:
        loaded_data = load_csv_data(file_path)

    if module == PackageList.py_impl:
        train_fn = model_train
    elif module == PackageList.c_pthread:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "c_implementations/c_pthread/dist/c_pthread-1.0-cp313-cp313-linux_x86_64.whl",
            ],
            check=True,
        )

        from c_pthread import train

        train_fn = train
    else:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "c_implementations/c_impl/dist/c_impl-1.0-cp313-cp313-linux_x86_64.whl",
            ],
            check=True,
        )

        from c_impl import train

        train_fn = train

    formatted_data = preprocess_loaded_data(loaded_data)

    train_all_dataset(formatted_data, train_fn)


if __name__ == "__main__":
    # You can set it to True if wanna see how it performs with raw kaggle data I found.
    # Certainly, you can also set url parameter if you found a same-alike data and want to test.
    # Additionally, if you want to use the c implementation (pthead enabled),
    # then you can set use_c_pthread  parameter to `True`!

    load_data_and_train_model(from_raw_csv=False, module=PackageList.c_pthread)
    # after initialization, the program is gonna serve the ui!
    serve_ui()

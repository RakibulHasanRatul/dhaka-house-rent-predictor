from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import (
    load_and_refit_kaggle_csv,
    preprocess_loaded_data,
)
from app.model.train import model_train, train_all_dataset
from app.serve import serve_ui
from config import FORMATTED_CSV_GIST_URL, RAW_KAGGLE_CSV_GIST_URL


def load_data_and_train_model(
    from_raw_csv: bool = False, url: str = "", use_c_pthread: bool = False
) -> None:
    if not url:
        url = RAW_KAGGLE_CSV_GIST_URL if from_raw_csv else FORMATTED_CSV_GIST_URL

    file_path = download_csv_from_gist(url)

    if from_raw_csv:
        loaded_data = load_and_refit_kaggle_csv(file_path)

    else:
        loaded_data = load_csv_data(file_path)

    if not use_c_pthread:
        train_fn = model_train
    else:
        import os
        import platform
        import subprocess

        system = platform.system()

        try:
            if system == "Windows":
                subprocess.run(
                    'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
                    shell=True,
                    check=True,
                )
            else:
                try:
                    subprocess.run(
                        "curl -LsSf https://astral.sh/uv/install.sh | sh",
                        shell=True,
                        check=True,
                    )
                except Exception:
                    subprocess.run(
                        "wget -qO- https://astral.sh/uv/install.sh | sh",
                        shell=True,
                        check=True,
                    )

            # Ensure uv is in the PATH.  This may require a restart of the shell
            # or explicitly setting the PATH.  For simplicity, we'll try adding
            # it to the current environment.  This won't persist, but might work.
            if system == "Windows":
                uv_path = os.path.join(os.environ["USERPROFILE"], ".uv", "bin")
            else:
                uv_path = os.path.join(os.environ["HOME"], ".uv", "bin")

            os.environ["PATH"] = uv_path + os.pathsep + os.environ["PATH"]

            subprocess.run(["uv", "pip", "install", "c_pthread/"], check=True)

            from c_pthread import train

            train_fn = train

        except subprocess.CalledProcessError as e:
            print(f"Error during uv installation or c_pthread installation: {e}")
            raise  # Re-raise the exception to halt execution

        except ImportError as e:
            print(f"Error importing c_pthread module: {e}")
            raise  # Re-raise the exception

    formatted_data = preprocess_loaded_data(loaded_data)

    train_all_dataset(formatted_data, train_fn)


if __name__ == "__main__":
    # You can set it to True if wanna see how it performs with raw kaggle data I found.
    # Certainly, you can also set url parameter if you found a same-alike data and want to test.
    # Additionally, if you want to use the c implementation (pthead enabled),
    # then you can set use_c_pthread  parameter to `True`!
    load_data_and_train_model(from_raw_csv=False, use_c_pthread=True)
    # after initialization, the program is gonna serve the ui!
    serve_ui()

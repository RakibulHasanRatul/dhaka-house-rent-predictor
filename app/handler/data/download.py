import os
import sys
import urllib.request as request

from config import DATA_DOWNLOAD_DIR


def flash_download_progress(
    total: int,
    downloaded: int,
    log: str = "\rDownloading: {bar} {percent:.2f}% ({downloaded}/{total} bytes)",
):
    bar_length = 25

    percent = (downloaded / total) * 100
    filled_length = int(bar_length * downloaded // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write(
        log.format(bar=bar, percent=percent, downloaded=downloaded, total=total)
    )
    sys.stdout.flush()


def download_csv_from_gist(url: str):
    with request.urlopen(url) as response:
        total_download_size = int(response.headers.get("Content-Length", 0))
        downloaded_total = 0
        block_size = 64 * 1024  # bytes
        # I think 64 Kibibytes is a descent block size! Isn't it?
        file_name = os.path.basename(url)
        file_path = f"{DATA_DOWNLOAD_DIR}/{file_name}"

        os.makedirs(DATA_DOWNLOAD_DIR, exist_ok=True)

        with open(file_path, "wb") as file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break  # obviously!
                # imagine it doesn't breaks here!

                file.write(buffer)
                downloaded_total += len(buffer)

                # using a helper function to organize the codeblock, nothing special here
                flash_download_progress(total_download_size, downloaded_total)

        sys.stdout.write("\n")

    print(f"\nFile downloaded to: {file_path} from url: {url}\n")

    return file_path

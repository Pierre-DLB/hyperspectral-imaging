import os
import yaml
import requests
from tqdm import tqdm


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("Content-Length", 0))

    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def get_data():
    with open("config_data.yaml", "r") as file:
        config = yaml.safe_load(file)

    base_url = config["url"]
    sources = ["IP", "PU", "KSC"]

    for source in sources:
        image_url = base_url[source]["original"]
        filename = f"{source}_original.mat"
        download_file(image_url, filename)


def get_labels():
    with open("config_data.yaml", "r") as file:
        config = yaml.safe_load(file)

    base_url = config["url"]
    sources = ["IP", "PU", "KSC"]

    for source in sources:
        file_url = base_url[source]["ground_truth"]
        filename = f"{source}_ground_truth.mat"
        download_file(file_url, filename)


if __name__ == "__main__":
    get_data()
    get_labels()

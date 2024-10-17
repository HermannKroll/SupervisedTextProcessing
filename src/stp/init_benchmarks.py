import os
import zipfile

from stp.config import BENCHMARKS_PATH

BENCHMARKS_TO_UNPACK = [
    "CDR.zip",
    "Chemprot.zip",
    "DDI.zip",

    "HallmarksOfCancer.zip",
    "LongCovid.zip",
    "Ohsumed.zip",
    "PharmaTech.zip"
]


def extract_benchmark_data():
    for file_name in BENCHMARKS_TO_UNPACK:
        file_path = os.path.join(BENCHMARKS_PATH, file_name)
        if not os.path.exists(file_path):
            print("Skipped {}".format(file_path))
            continue

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(BENCHMARKS_PATH)
        print("Extracted {} into {}".format(file_name, BENCHMARKS_PATH))


if __name__ == "__main__":
    extract_benchmark_data()

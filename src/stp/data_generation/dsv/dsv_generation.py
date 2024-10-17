import os

import pandas as pd

from stp.benchmark import BenchmarkLabel
from stp.config import DSV_SENTENCES_PATH

BENCHMARK_RUN = "train"


class DSVDataGeneration:
    def __init__(self, data_source, run=BENCHMARK_RUN):
        assert run in {"train"}

        self.run = run
        self.data_source = data_source
        self._load_data()

    def __repr__(self):
        return f"DSV Generator: {self.data_source}.{self.run}: [{self.negatives()} (-)|{self.positives()} (+)]"

    def __str__(self):
        return self.__repr__()

    def _load_data(self):
        path = os.path.join(DSV_SENTENCES_PATH, f"{self.data_source}.relabeled.csv")
        assert os.path.exists(path)
        df = pd.read_csv(path)
        self.sentences = df

    def negatives(self):
        return len(self.sentences[self.sentences["label"] == BenchmarkLabel.NEGATIVE])

    def positives(self):
        return len(self.sentences[self.sentences["label"] != BenchmarkLabel.NEGATIVE])


if __name__ == "__main__":
    gen = DSVDataGeneration("train")
    print(gen)

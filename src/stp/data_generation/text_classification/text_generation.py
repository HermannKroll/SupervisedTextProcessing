import ast
import os

import pandas as pd

from stp.config import HALLMARKS_OF_CANCER_PATH, LONG_COVID_PATH, OHSUMED_PATH, PHARMA_TECH_PATH


TEXT_BENCHMARKS = {
    "HallmarksOfCancer": HALLMARKS_OF_CANCER_PATH,
    "LongCovid": LONG_COVID_PATH,
    "Ohsumed": OHSUMED_PATH,
    "PharmaTech": PHARMA_TECH_PATH,
}


def parse_hallmarks_of_cancer_labels(label) -> list:
    assert isinstance(label, str)
    return list(int(l) for l in ast.literal_eval(label))


def parse_long_covid_label(label) -> list:
    assert isinstance(label, int)
    return [label]


def parse_ohsumed_label(label) -> list:
    assert isinstance(label, str)
    labels = label.split(",")
    return list(int(l.strip()) for l in labels)


def parse_pharma_tech_label(label) -> list:
    assert isinstance(label, int)
    return [label]


def transform_text_label(benchmark: str, label: int, to_intern=True):
    if benchmark == "HallmarksOfCancer":
        factor = 0
    elif benchmark == "LongCovid":
        factor = 1 * 100
    elif benchmark == "Ohsumed":
        factor = 2 * 100
    elif benchmark == "PharmaTech":
        factor = 3 * 100
    else:
        raise Exception("Unknown benchmark")

    if to_intern:
        return label + factor
    else:
        return label - factor


TEXT_BENCHMARKS_LABEL_PARSER = {
    "HallmarksOfCancer": parse_hallmarks_of_cancer_labels,
    "LongCovid": parse_long_covid_label,
    "Ohsumed": parse_ohsumed_label,
    "PharmaTech": parse_pharma_tech_label
}

BENCHMARK_RUN = "train"


class TextDataGeneration:
    def __init__(self, data_source, run=BENCHMARK_RUN):
        # translate to internal description
        if run == "dev":
            run = "val"

        assert run in {"test", "train", "val"}

        self.run = run
        self.data_source = data_source
        self._load_data()

    def __repr__(self):
        classes = ' | '.join((str(k) + ": " + str(v)) for k, v in self.class_distribution().items())
        return f"TextDataGenerator: {self.data_source}.{self.run}: [{classes}]"

    def __str__(self):
        return self.__repr__()

    def _load_data(self):
        path = os.path.join(TEXT_BENCHMARKS[self.data_source], f"{self.run}.tsv")
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels"])
        rows = list()
        parse_label = TEXT_BENCHMARKS_LABEL_PARSER[self.data_source]
        doc_id = 0
        for text, labels in zip(df["text"], df["labels"]):
            labels = parse_label(labels)
            for label in labels:
                rows.append((text, transform_text_label(self.data_source, label), doc_id))
                doc_id += 1
        self.sentences = pd.DataFrame(rows, columns=["text", "label", "id"])

    def class_distribution(self):
        label_distribution = dict()

        for label in self.sentences["label"]:
            if label not in label_distribution:
                label_distribution[label] = 0
            label_distribution[label] += 1

        return label_distribution


if __name__ == "__main__":
    for benchmark, _ in TEXT_BENCHMARKS.items():
        for benchmark_run in ["test", "train", "val"]:
            gen = TextDataGeneration(benchmark, benchmark_run)
            print(gen)

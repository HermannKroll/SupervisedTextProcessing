import enum
from typing import List

from stp.document import parse_cdr_run, parse_chemprot_run, parse_ddi_run, Sentence

BENCHMARK_TO_RUN_PATH = {
    "CDR": {
        "train": "CDR_TrainingSet.PubTator.txt",
        "test": "CDR_TestSet.PubTator.txt",
        "dev": "CDR_DevelopmentSet.PubTator.txt",
        "parse_function": parse_cdr_run
    },
    "Chemprot": {
        "train": "training",
        "test": "test",
        "dev": "development",
        "parse_function": parse_chemprot_run
    },
    "DDI": {
        "train": "Train",
        "test": "Test",
        "parse_function": parse_ddi_run
    }
}

BENCHMARK_TO_NAME = {
    "CDR": "ChemicalDiseaseRelationBenchmark",
    "Chemprot": "ChemicalProteinRelationBenchmark",
    "DDI": "DrugDrugInteractionBenchmark"
}


class BenchmarkLabel(enum.IntEnum):
    NEGATIVE = 0
    CDR = 1
    CHEMPROT = 2
    DDI = 3

    @staticmethod
    def to_value(run: str):
        if run == "CDR":
            return BenchmarkLabel.CDR
        if run == "Chemprot":
            return BenchmarkLabel.CHEMPROT
        if run == "DDI":
            return BenchmarkLabel.DDI
        raise NotImplementedError("Unknown run")


class ChemprotLabel(enum.IntEnum):
    NEGATIVE = 0
    UPREGULATOR = 3
    DOWNREGULATOR = 4

    @staticmethod
    def to_value(run: str):
        if run == "UPREGULATOR":
            return ChemprotLabel.UPREGULATOR
        if run == "DOWNREGULATOR":
            return ChemprotLabel.DOWNREGULATOR
        raise NotImplementedError("Unknown run")


class MultitaskLabel(enum.IntEnum):
    NEGATIVE = 0
    CDR = 1
    DDI = 2
    UPREGULATOR = 3
    DOWNREGULATOR = 4

    @staticmethod
    def to_value(run: str):
        if run == "CDR":
            return MultitaskLabel.CDR
        if run == "DDI":
            return MultitaskLabel.DDI
        if run == "UPREGULATOR":
            return MultitaskLabel.UPREGULATOR
        if run == "DOWNREGULATOR":
            return MultitaskLabel.DOWNREGULATOR
        raise NotImplementedError("Unknown run")


class Benchmark:
    def __init__(self, name):
        assert name in set(BENCHMARK_TO_RUN_PATH.keys()), "Invalid benchmark name"

        self.name = name
        self.documents = dict()
        self.sentences = dict()
        self.__load_documents()
        self.__prepare_sentences()

        print(f"Benchmark {self.name} initialized "
              f"(Test: {len(self.documents['test'])}, {len(self.sentences['test'])}, "
              f"Train: {len(self.documents['train'])}, {len(self.sentences['train'])}, "
              f"Dev: {len(self.documents['dev']) if name != 'DDI' else 0}, "
              f"{len(self.sentences['dev']) if name != 'DDI' else 0})")

    def __load_documents(self):
        parse_function = BENCHMARK_TO_RUN_PATH[self.name]["parse_function"]

        train_path = BENCHMARK_TO_RUN_PATH[self.name]["train"]
        self.documents["train"] = parse_function(train_path)
        test_path = BENCHMARK_TO_RUN_PATH[self.name]["test"]
        self.documents["test"] = parse_function(test_path)
        if "dev" in BENCHMARK_TO_RUN_PATH[self.name]:
            dev_path = BENCHMARK_TO_RUN_PATH[self.name]["dev"]
            self.documents["dev"] = parse_function(dev_path)

    def __prepare_sentences(self):
        for run, documents in self.documents.items():
            sentences = list()
            for document in documents:
                sentences.extend(document.get_sentences())
            self.sentences[run] = sentences

    def get_train_sentences(self) -> List[Sentence]:
        return self.sentences["train"]

    def get_test_sentences(self) -> List[Sentence]:
        return self.sentences["test"]

    def get_dev_sentences(self) -> List[Sentence]:
        return self.sentences["dev"]

    def __str__(self):
        return (f"<Benchmark {BENCHMARK_TO_NAME[self.name]} "
                f"Training({len(self.documents['train'])}) "
                f"Test({len(self.documents['test'])})>")

    def __repr__(self):
        return self.__str__()

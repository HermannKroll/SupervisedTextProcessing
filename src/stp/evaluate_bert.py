import datetime
import json
import os
from collections import defaultdict

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
from matplotlib import pyplot as plt

from stp.benchmark import BenchmarkLabel, MultitaskLabel
from stp.config import BENCHMARK_RESULTS_PATH, DIAGRAMS_PATH
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.chemprot_generation import ChemprotDataGeneration
from stp.data_generation.text_classification.text_generation import transform_text_label, TextDataGeneration
from stp.prediction.bert_pipeline import BertPredictionPipeline
from stp.run_config import MODEL_TYPE, BERT_MODEL_PATH, TASK, NUMBER_OF_PUBMED_DOCUMENTS, \
    AVG_SENTENCES_PER_DOCUMENT, METHOD_TO_PRINT_NAME
from stp.training.GeneralModelTrainerBase import format_time
from stp.util.dataframe import reduce_dataframe


BERT_MODELS = [
    "bert",
    "bert_worst",
    "bert_default",
    "roberta",
    "roberta_worst",
    "roberta_default",
    "xlnet",
    "xlnet_worst",
    "xlnet_default",
    "biobert",
    "biobert_worst",
    "biobert_default",
    "biolinkbert",
    "biolinkbert_worst",
    "biolinkbert_default",
    "pubmedbert",
    "pubmedbert_worst"
    "pubmedbert_default"
]

BERT_MODELS_2_NAME = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "xlnet": "XLNet",
    "biolinkbert": "BioLinkBERT",
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
}

BENCHMARKS_RE = [
    "CDR",
    "Chemprot_c",
    "Chemprot",
    "DDI"
]

BENCHMARKS_TC = [
    "HallmarksOfCancer",
    "Ohsumed",
    "LongCovid",
    "PharmaTech",
]

TIME_MEASURE_MIN_PER_CLASS = 500_000


def get_data_generator(benchmark: str, time_measurement=False):
    if time_measurement:
        if MODEL_TYPE == "multi":
            raise NotImplementedError("Time measurement is not implemented for multiclass")
        if benchmark == "Chemprot_c":
            gen_test = ChemprotDataGeneration("test", verbose=False)
            gen_train = ChemprotDataGeneration("train", verbose=False)
            gen_dev = ChemprotDataGeneration("dev", verbose=False)
        elif TASK == "TC":
            gen_test = TextDataGeneration(benchmark, "test")
            gen_train = TextDataGeneration(benchmark, "train")
            gen_dev = TextDataGeneration(benchmark, "dev")
        else:
            gen_test = BaselineDataGeneration(benchmark, "test", verbose=False)
            gen_train = BaselineDataGeneration(benchmark, "train", verbose=False)
            gen_dev = BaselineDataGeneration(benchmark, "dev", verbose=False)

        df = pd.concat([gen_test.sentences, gen_train.sentences, gen_dev.sentences])

        # simulate large dataset by sampling up the data
        labels = df["label"].unique()
        while True:
            if all(len(df.loc[df["label"] == l]) >= TIME_MEASURE_MIN_PER_CLASS for l in labels):
                break

            df = df.loc[df.index.repeat(2)].reset_index(drop=True)

        dfs = list()
        for l in labels:
            df_class = reduce_dataframe(df.loc[df["label"] == l], TIME_MEASURE_MIN_PER_CLASS)
            dfs.append(df_class)

        df = pd.concat(dfs)
        assert all(len(df.loc[df["label"] == l]) == TIME_MEASURE_MIN_PER_CLASS for l in labels)

        gen_test.sentences = df
        return gen_test

    if benchmark == "Chemprot_c":
        gen = ChemprotDataGeneration("test", verbose=False)
    elif TASK == "TC":
        gen = TextDataGeneration(benchmark, "test")
        if MODEL_TYPE == "multi":
            benchmark_intern_label = transform_text_label(benchmark, 0)
            gen.sentences["label"].replace(benchmark_intern_label, 0, inplace=True)
    else:
        gen = BaselineDataGeneration(benchmark, "test", verbose=False)
        if MODEL_TYPE == "multi":
            gen.sentences["label"].replace(BenchmarkLabel.to_value(benchmark),
                                           MultitaskLabel.to_value(benchmark),
                                           inplace=True)
    return gen


def run_test_classification_bert(device="cpu", time_measurement=False):
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    start = datetime.datetime.now()
    lbp = BertPredictionPipeline()
    for benchmark in benchmarks:
        print("Benchmark:", benchmark)
        if benchmark == "Chemprot" and MODEL_TYPE == "multi":
            continue
        if benchmark == "Chemprot_c" and MODEL_TYPE == "dsv":
            continue

        gen = get_data_generator(benchmark, time_measurement=time_measurement)

        dataset = BERT_MODEL_PATH
        if MODEL_TYPE == "multi":
            model_folder = dataset
        else:
            model_folder = os.path.join(dataset, benchmark)

        if not os.path.exists(model_folder):
            print("Skipping method", model_folder)
            continue

        for bert_model in BERT_MODELS:
            if not os.path.exists(os.path.join(model_folder, f"{bert_model}_model")):
                print("skipping BERT model", model_folder, bert_model)
                continue

            round_start = datetime.datetime.now()

            desc = f"{benchmark}.{dataset.split('/')[-1]}.{bert_model}"
            print(desc, end=": ", flush=True)

            predictions, scores = lbp.predict_label(gen.sentences, model_folder, bert_model,
                                                    device=device)
            assert len(predictions) == len(gen.sentences["id"])

            result_df = pd.DataFrame()
            result_df["id"] = gen.sentences["id"]
            result_df["prediction"] = predictions

            desc = f"{MODEL_TYPE}.{benchmark}.{bert_model}"
            if device == "cuda":
                desc += ".gpu"

            if not time_measurement:
                out_path = os.path.join(BENCHMARK_RESULTS_PATH, f"{desc}.csv")
                result_df.to_csv(out_path)
            else:
                desc += ".time_measurement"

            out_path = os.path.join(BENCHMARK_RESULTS_PATH, f"{desc}.statistics.json")
            with open(out_path, "w") as outfile:
                json.dump(scores, outfile, indent=2)

            print(datetime.datetime.now() - round_start)

    print("Test-Evaluation finished after:", datetime.datetime.now() - start)


def evaluate_results(device="cpu"):
    rows = defaultdict(list)
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:

        for classifier in BERT_MODELS:
            if classifier not in BERT_MODELS_2_NAME:
                continue

            if device == "cuda":
                file_name = f"{MODEL_TYPE}.{benchmark}.{classifier}.gpu.statistics.json"
            else:
                file_name = f"{MODEL_TYPE}.{benchmark}.{classifier}.statistics.json"
            file_path = os.path.join(BENCHMARK_RESULTS_PATH, file_name)

            if not os.path.isfile(file_path):
                p = "-"
                r = "-"
                f1 = "-"
            else:
                with open(file_path) as f:
                    statistics = json.load(f)

                p = round(statistics['precision'], 2)
                r = round(statistics['recall'], 2)
                f1 = round(statistics['f1_score'], 2)

            prefix = BERT_MODELS_2_NAME[classifier]

            rows[prefix].extend([p, r, f1])

    for prefix, values in rows.items():
        assert len(values) == len(benchmarks) * 3, f"{prefix} {len(values)}"
        print(f"{prefix} &", " & ".join(str(r) for r in values), r" \\")


def application_time(device="cpu", time_measurement=False):
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:
        for classifier in BERT_MODELS:
            if classifier not in BERT_MODELS_2_NAME:
                continue

            file_name = f"{MODEL_TYPE}.{benchmark}.{classifier}"
            if device == "cuda":
                file_name += ".gpu"
            if time_measurement:
                file_name += ".time_measurement"
            file_name += ".statistics.json"
            file_path = os.path.join(BENCHMARK_RESULTS_PATH, file_name)

            with open(file_path) as f:
                statistics = json.load(f)

            prefix = BERT_MODELS_2_NAME[classifier]

            time_taken = statistics["time_taken"]
            sentences = statistics["sentences"]

            tps = time_taken / sentences
            factor = (AVG_SENTENCES_PER_DOCUMENT * NUMBER_OF_PUBMED_DOCUMENTS if TASK == "RE"
                      else NUMBER_OF_PUBMED_DOCUMENTS)
            tpp = factor * tps

            time_per_sentence_string = f"{tps:0.2e} s"
            time_per_pubmed_string = format_time(tpp)

            print(f"{benchmark:<18}", f"{prefix:<15}", f"{sentences:>6}", time_per_sentence_string, "&",
                  time_per_pubmed_string)


def evaluate_bert_results_graphically():
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in list(benchmarks):
        data = defaultdict(lambda: defaultdict(list))
        columns = ["Method"]
        for classifier in BERT_MODELS:
            columns.append(classifier.title())

        for classifier in BERT_MODELS:
            file_name = f"{MODEL_TYPE}.{benchmark}.{classifier}.json"

            file_path = os.path.join(BENCHMARK_RESULTS_PATH, file_name)
            if not os.path.isfile(file_path):
                print("skipped", file_name)

            with open(file_path) as f:
                statistics = json.load(f)

            name = METHOD_TO_PRINT_NAME["binaryclass"]
            # print(file_name, precision, recall, f1_score)
            data["Precision"][name].append(statistics.get("precision", 0.0))
            data["Recall"][name].append(statistics.get("recall", 0.0))
            data["F1 Score"][name].append(statistics.get("f1_score", 0.0))

        for metric in ["Precision", "Recall", "F1 Score"]:
            score_data = list()

            for method_name, scores in data[metric].items():
                score_data.append((method_name, *scores))

            df = pd.DataFrame(score_data, columns=columns)
            df.set_index("Method", inplace=True)
            ax: plt.Axes = df.plot(kind="bar", grid=True, ylim=(0, 1))

            ax.tick_params(axis='x', labelrotation=45)
            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed')

            ax.legend(loc="upper left", ncols=2, fontsize="small")

            plt.ylabel(metric)
            plt.title(benchmark)
            plt.subplots_adjust(bottom=0.25, left=0.15)

            file_name = f"{MODEL_TYPE}.{benchmark}.{metric.lower()}.bert.png"
            plt.savefig(os.path.join(DIAGRAMS_PATH, file_name))
            plt.close()


if __name__ == "__main__":
    run_test_classification_bert(device="cpu", time_measurement=False)
    # evaluate_results()
    # application_time()
    # evaluate_bert_results_graphically()

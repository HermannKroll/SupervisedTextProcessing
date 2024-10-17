import datetime
import json
import os
from collections import defaultdict

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
from matplotlib import pyplot as plt

from stp.config import BENCHMARK_RESULTS_PATH, DIAGRAMS_PATH
from stp.evaluate_bert import get_data_generator
from stp.prediction.pipeline import PredictionPipeline
from stp.run_config import CLASSIFIER, VECTORIZER, MODEL_TYPE, MODEL_VARIANT, \
    METHOD_TO_PRINT_NAME, MODEL_VARIANT_TO_PRINT_NAME, TASK, AVG_SENTENCES_PER_DOCUMENT, NUMBER_OF_PUBMED_DOCUMENTS
from stp.training.GeneralModelTrainerBase import format_time

BENCHMARKS_RE = [
    "CDR",
    "Chemprot_c",
    "Chemprot",
    "DDI"
]

BENCHMARKS_TC = [
    "HallmarksOfCancer",
    "LongCovid",
    "Ohsumed",
    "PharmaTech",
]


def run_test_classification_traditional(time_measurement=False):
    start = datetime.datetime.now()
    lbp = PredictionPipeline()
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:

        if benchmark == "Chemprot" and MODEL_TYPE == "multi":
            continue

        if benchmark == "Chemprot_c" and MODEL_TYPE == "dsv":
            continue

        print("Benchmark:", benchmark)
        gen = get_data_generator(benchmark, time_measurement=time_measurement)

        for vector_model in VECTORIZER.values():

            base_path = vector_model
            if not os.path.exists(base_path):
                print("Skipping method", base_path)
                continue

            for classifier in CLASSIFIER:
                for model_name, model_description in MODEL_VARIANT.items():
                    if MODEL_TYPE == "multi":
                        model_path = os.path.join(str(base_path), "multiclass", f"{classifier}_{model_name}.joblib")
                    else:
                        model_path = os.path.join(str(base_path), benchmark, f"{classifier}_{model_name}.joblib")

                    if not os.path.exists(model_path):
                        print("Skipping classifier", model_path)
                        continue

                    round_start = datetime.datetime.now()

                    vector_model_name = vector_model.split('/')[-1]
                    desc = (f"{MODEL_TYPE}.{benchmark}.{vector_model_name}.{classifier}."
                            f"{model_description}")
                    print(desc, end=": ", flush=True)

                    predictions, scores = lbp.predict_label(gen.sentences, model_path)
                    assert len(predictions) == len(gen.sentences["id"])

                    result_df = pd.DataFrame()
                    result_df["id"] = gen.sentences["id"]
                    result_df["prediction"] = predictions

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


CLASSIFIER_TO_PRINT_NAME = {
    "svc": "SVC",
    "rf": "Random Forrest",
    "xgb": "XGBoost"
}

EMBEDDING_TO_PRINT_NAME = {
    "tfidf": "tfidf",
    "sbert": "sBERT"
}


def evaluate_results():
    rows = defaultdict(list)
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC

    for benchmark in benchmarks:
        for idx_c, classifier in enumerate(CLASSIFIER):
            for idx_e, embedding in enumerate(VECTORIZER):
                model_description = "tuned_params"

                file_name = (f"{MODEL_TYPE}.{benchmark}.{embedding}.{classifier}.{model_description}"
                             f".statistics.json")
                file_path = os.path.join(BENCHMARK_RESULTS_PATH, file_name)

                if not os.path.isfile(file_path):
                    print("skipped", file_path)
                    p = "-"
                    r = "-"
                    f1 = "-"
                else:
                    with open(file_path) as f:
                        statistics = json.load(f)

                    p = round(statistics['precision'], 2)
                    r = round(statistics['recall'], 2)
                    f1 = round(statistics['f1_score'], 2)

                prefix = CLASSIFIER_TO_PRINT_NAME[classifier] + " + " + EMBEDDING_TO_PRINT_NAME[embedding]

                rows[prefix].extend([p, r, f1])

    for prefix, values in rows.items():
        assert len(values) == 12, f"{prefix} {len(values)}"
        print(f"{prefix} &", " & ".join(str(r) for r in values), r" \\")


def application_time():
    benchmark = "DDI"

    for classifier in CLASSIFIER:
        for embedding in VECTORIZER:
            model_description = "tuned_params"

            file_name = f"{MODEL_TYPE}.{benchmark}.{embedding}.{classifier}.{model_description}.statistics.json"
            file_path = os.path.join(BENCHMARK_RESULTS_PATH, file_name)

            with open(file_path) as f:
                statistics = json.load(f)

            prefix = CLASSIFIER_TO_PRINT_NAME[classifier] + " + " + EMBEDDING_TO_PRINT_NAME[embedding]

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


def evaluate_results_graphically():
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:
        data = defaultdict(lambda: defaultdict(list))
        columns = ["Method"]
        for classifier in CLASSIFIER:
            for model_description in MODEL_VARIANT.values():
                columns.append(classifier.upper() + " + " + MODEL_VARIANT_TO_PRINT_NAME[model_description])

        for classifier in CLASSIFIER:
            embeddings = VECTORIZER.keys()

            for embedding in embeddings:
                for model_description in MODEL_VARIANT.values():
                    file_name = (f"{MODEL_TYPE}.{benchmark}.{embedding}.{classifier}.{model_description}"
                                 f".statistics.json")
                    file_path = os.path.join(BENCHMARK_RESULTS_PATH, file_name)

                    if not os.path.isfile(file_path):
                        print("skipped", file_name)

                    with open(file_path) as f:
                        statistics = json.load(f)

                    name = embedding + "\n" + METHOD_TO_PRINT_NAME["binaryclass"]
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

            file_name = f"{MODEL_TYPE}.{benchmark}.{metric.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(DIAGRAMS_PATH, file_name))
            plt.close()


if __name__ == "__main__":
    run_test_classification_traditional(time_measurement=True)
    # evaluate_results()
    # application_time()
    # evaluate_results_graphically()

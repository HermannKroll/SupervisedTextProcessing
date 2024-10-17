import json
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from stp.config import TC_NOICE_EVAL_PATH, BENCHMARK_RESULTS_PATH, TC_NOISE_PATH
from stp.data_generation.text_classification.text_generation import TextDataGeneration
from stp.diagrams import BENCHMARKS_RE, BENCHMARKS_TC
from stp.prediction.bert_pipeline import BertPredictionPipeline
from stp.prediction.pipeline import PredictionPipeline
from stp.run_config import TASK, MODEL_TYPE, TC_FLIPPING_VALUES, \
    BERT_MODEL_PATH_MAPPING, MODEL_TYPE_MAPPING, TC_REDUCTION_VALUES


def evaluate_noise_traditional(model_type):
    assert model_type in {"flipping", "reduction"}
    base_paths = MODEL_TYPE_MAPPING.get(model_type)
    lbp = PredictionPipeline()
    num_configs = TC_REDUCTION_VALUES
    models = ["svc", "xgb"]
    vectorizer = "tfidf"

    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:
        gen = TextDataGeneration(benchmark, "test")
        base_path = base_paths.get(vectorizer)
        for model in models:
            for i in num_configs:
                benchmark_i = benchmark + f"_{int(i * 100)}"
                model_file = f"{model}_model.joblib"
                model_path = os.path.join(base_path, benchmark_i, model_file)

                desc = f"{model_type}.{benchmark}.{vectorizer}.{model}.{int(i * 100)}"

                if not os.path.exists(model_path):
                    print("skipped", desc)
                    continue

                if os.path.exists(os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.statistics.json")):
                    print("existing", desc)
                    continue

                round_start = time.time()
                print(desc, i, end=": ", flush=True)
                predictions, probabilities, scores = lbp.predict_label(gen.sentences, model_path)
                assert len(predictions) == len(probabilities) == len(gen.sentences["id"])

                result_df = pd.DataFrame()
                result_df["id"] = gen.sentences["id"]
                result_df["prediction"] = predictions
                result_df["probability"] = probabilities

                out_path = os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.csv")
                result_df.to_csv(out_path)

                out_path = os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.statistics.json")
                with open(out_path, "w") as outfile:
                    json.dump(scores, outfile, indent=2)

                print(time.time() - round_start)


def evaluate_noise_bert(model_type):
    assert model_type in {"flipping", "reduction"}
    base_path = BERT_MODEL_PATH_MAPPING.get(model_type)
    models = ["biobert", "pubmedbert"]
    lbp = BertPredictionPipeline()
    benchmarks = BENCHMARKS_TC
    num_configs = TC_FLIPPING_VALUES
    for benchmark in benchmarks:
        gen = TextDataGeneration(benchmark, "test")
        for model in models:
            for i in num_configs:
                benchmark_i = benchmark + f"_{int(i * 100)}"
                model_file = f"{model}_model"
                model_path = os.path.join(base_path, benchmark_i)

                desc = f"{model_type}.{benchmark}.{model}.{int(i * 100)}"

                if not os.path.exists(os.path.join(model_path, model_file)):
                    print("skipped", desc)
                    continue

                if os.path.exists(os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.statistics.json")):
                    print("existing", desc)
                    continue

                round_start = time.time()
                print(desc, i, end=": ", flush=True)
                predictions, probabilities, scores = lbp.predict_label(gen.sentences, model_path, model_file,
                                                                       device="cuda")
                assert len(predictions) == len(probabilities) == len(gen.sentences["id"])

                result_df = pd.DataFrame()
                result_df["id"] = gen.sentences["id"]
                result_df["prediction"] = predictions
                result_df["probability"] = probabilities

                out_path = os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.csv")
                result_df.to_csv(out_path)

                out_path = os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.statistics.json")
                with open(out_path, "w") as outfile:
                    json.dump(scores, outfile, indent=2)

                print(time.time() - round_start)


MODEL_2_PRINT_NAME = {
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
    "tfidf.xgb": "XGBoost + tfidf",
    "tfidf.svc": "SVC + sBERT"
}


LINE_STYLES = [
    "solid",
    "dotted",
    "dashed",
    "dashdot"
]

MARKER_STYLES = [
    "^",
    "v",
    "+",
    "x",
]


def draw_diagram(model_type):
    assert model_type in {"flipping", "reduction"}
    if model_type == "flipping":
        num_configs = [0.0] + TC_FLIPPING_VALUES
        xlabel = "Fraction of the flipped data"
    else:
        num_configs = TC_REDUCTION_VALUES + [1.0]
        xlabel = "Fraction of the full dataset"
    benchmarks = BENCHMARKS_TC
    models = ["tfidf.svc", "tfidf.xgb", "biobert", "pubmedbert"]

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(22, 4))
    for index, (benchmark, bm_name) in enumerate(benchmarks.items()):
        ax: plt.Axes = axes[index]
        for style_index, model in enumerate(models):
            results = list()
            for i in num_configs:
                if i == 0.0 or i == 1.0:
                    desc = f"baseline.{benchmark}.{model}"
                    if "tfidf" in model:
                        statistics_path = os.path.join(BENCHMARK_RESULTS_PATH, f"{desc}.tuned_params.statistics.json")
                    else:
                        statistics_path = os.path.join(BENCHMARK_RESULTS_PATH, f"{desc}.gpu.statistics.json")

                else:
                    desc = f"{model_type}.{benchmark}.{model}.{int(i * 100)}"
                    statistics_path = os.path.join(TC_NOICE_EVAL_PATH, f"{desc}.statistics.json")

                if not os.path.exists(statistics_path):
                    print("skipped", statistics_path)
                    results.append(np.nan)
                    continue

                with open(statistics_path, "rt") as f:
                    statistics = json.load(f)

                results.append(statistics.get("f1_score"))
            ax.plot(num_configs,
                    results,
                    linestyle=LINE_STYLES[style_index],
                    # marker=MARKER_STYLES[style_index],
                    label=MODEL_2_PRINT_NAME[model])
        ax.set_title(bm_name)
        ax.set(ylim=(0, 1),
               xticks=num_configs,
               xticklabels=num_configs,
               ylabel="F1-Score",
               xlabel=xlabel)
        ax.legend()

    plt.savefig(os.path.join(TC_NOISE_PATH, f"{model_type}.png"), dpi=300)


if __name__ == "__main__":
    assert TASK == "TC"
    # evaluate_noise_traditional("reduction")
    # evaluate_noice_bert("reduction")
    draw_diagram("reduction")

    # evaluate_noise_traditional("flipping")
    # evaluate_noise_bert("flipping")
    draw_diagram("flipping")

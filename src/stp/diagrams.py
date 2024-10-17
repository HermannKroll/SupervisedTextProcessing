import glob
import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from stp.config import BOXPLOT_PATH, HYPERPARAMETER_SEARCH_PATH, CLASS_DISTRIBUTION_PATH
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.chemprot_generation import ChemprotDataGeneration
from stp.data_generation.text_classification.text_generation import TextDataGeneration, transform_text_label
from stp.run_config import TASK

BEST_MODELS_RE = {
    "shallow": {
        "first": ["tfidf.svc", "SVC + tfidf"],
        "second": ["tfidf.xgb", "XGBoost + tfidf"]
    },
    "lm": {
        "first": ["biolinkbert", "BioLinkBERT"],
        "second": ["pubmedbert", "PubMedBERT"]
    }
}

BEST_MODELS_TC = {
    "shallow": {
        "first": ["tfidf.svc", "SVC + tfidf"],
        "second": ["tfidf.xgb", "XGBoost + tfidf"]
    },
    "lm": {
        "first": ["pubmedbert", "PubMedBERT"],
        "second": ["biobert", "BioBERT"]
    }
}

BENCHMARKS_RE = {
    "CDR": "CDR",
    "Chemprot_c": "ChemProtC",
    "Chemprot": "ChemProtE",
    "DDI": "DDI"
}

BENCHMARKS_TC = {
    "PharmaTech": "Pharmaceutical Technology",
    "HallmarksOfCancer": "Hallmarks of Cancer",
    "Ohsumed": "OHSUMED",
    "LongCovid": "Long Covid"
}


def draw_box_plot(metric="accuracy"):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    best_models = BEST_MODELS_RE if TASK == "RE" else BEST_MODELS_TC
    for idx, (benchmark, benchmark_name) in enumerate(benchmarks.items()):
        results = dict()
        format_path = f"baseline.{benchmark}.{{}}.*.statistics.json"

        for strategy, best_two in best_models.items():
            for model, config in best_two.items():
                model_config, name = config

                model_config = format_path.format(model_config)
                model_paths = os.path.join(HYPERPARAMETER_SEARCH_PATH, model_config)
                model_paths = glob.glob(model_paths)

                results[name] = list()
                for model_path in model_paths:
                    with open(model_path, "rt") as f:
                        data = json.load(f)
                    results[name].append(data[metric])

        max_len = max(len(v) for v in results.values())
        for name, result in results.items():
            if "SVC + tfidf" in name and len(result) < 32:
                result.clear()
                result.extend([np.nan] * 36)
            if len(result) < max_len:
                result.extend([np.nan] * (max_len - len(result)))

        df = pd.DataFrame.from_dict(results, orient="columns")

        ax = axes.flat[idx]
        df.boxplot(ax=ax, rot=45)
        ax.set_ylim([0, 1])
        ax.get_xaxis().offsetText.set_visible(False)
        ax.get_yaxis().set_label_text("Accuracy")
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title(benchmark_name)

    plt.tight_layout()
    plt.savefig(os.path.join(BOXPLOT_PATH, f"{TASK}_HS_Comparison.png"))


BENCHMARK_TO_LABEL = {
    "CDR": {0: "Negative", 1: "Positive"},
    "Chemprot_c": {0: "Negative", 3: "Up", 4: "Down"},
    "Chemprot": {0: "Negative", 2: "Positive"},
    "DDI": {0: "Negative", 3: "Positive"},
}


def draw_class_distribution():
    fig, axes = plt.subplots(nrows=2, ncols=2)
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for idx, (benchmark, name) in enumerate(benchmarks.items()):
        if TASK == "RE":
            if benchmark == "Chemprot_c":
                gen = ChemprotDataGeneration("train", verbose=False)
            else:
                gen = BaselineDataGeneration(benchmark, "train", verbose=False)
            sentences = gen.sentences

            label_dict = BENCHMARK_TO_LABEL[benchmark]

            label_count = {v: 0 for v in label_dict.values()}
            for _, _, label, _ in sentences.itertuples(index=False, name=None):
                label_count[label_dict[label]] += 1
        else:
            gen = TextDataGeneration(benchmark, "train")
            label_distr = gen.class_distribution()
            label_count = {}
            for key, value in label_distr.items():
                label = f"Label {transform_text_label(benchmark, key, to_intern=False)}"
                label_count[label] = value
            label_count = dict(sorted(label_count.items(), key=lambda item: item[1], reverse=True))

        keys, values = zip(*label_count.items())
        keys = [k.split(" ")[-1].strip() for k in keys]
        df = pd.DataFrame({"Labels": keys, "Relations": values})
        ax = axes.flat[idx]
        if TASK == "RE":
            df.plot.bar(ax=ax, x='Labels', y='Relations', rot=0, xlabel="Label", ylabel="Sample size")
        else:
            df.plot.bar(ax=ax, x='Labels', y='Relations', rot=90, xlabel="Label", ylabel="Sample size")
            ax.tick_params(axis='x', labelsize=7)
        ax.get_legend().remove()
        ax.grid(visible=True)
        ax.set_axisbelow(True)
        ax.set_title(name)

        print(benchmark)
        print(df)

    plt.tight_layout()
    plt.savefig(os.path.join(CLASS_DISTRIBUTION_PATH, f"{TASK}_Label_Distribution.png"), dpi=300)


if __name__ == "__main__":
    draw_box_plot()
    draw_class_distribution()

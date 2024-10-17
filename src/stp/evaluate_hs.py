import json
import os
import time

import pandas as pd

from stp.config import HYPERPARAMETER_SEARCH_PATH
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.chemprot_generation import ChemprotDataGeneration
from stp.data_generation.text_classification.text_generation import TextDataGeneration
from stp.evaluate_traditional import BENCHMARKS_RE, BENCHMARKS_TC
from stp.prediction.bert_pipeline import BertPredictionPipeline
from stp.prediction.pipeline import PredictionPipeline
from stp.run_config import TASK, VECTORIZER, MODEL_TYPE, BERT_MODEL_PATH


def evaluate_hyperparameter_search_traditional():
    base_paths = VECTORIZER.items()
    models = ["svc"] # + ["xgb"]
    lbp = PredictionPipeline()
    num_configs = 32

    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:

        if TASK == "TC":
            gen = TextDataGeneration(benchmark, "test")
        elif TASK == "RE":
            if benchmark == "Chemprot_c":
                gen = ChemprotDataGeneration("test")
            else:
                gen = BaselineDataGeneration(benchmark, "test")
        else:
            raise NotImplementedError()

        for vectorizer, base_path in base_paths:
            for model in models:
                for i in range(0, num_configs):
                    model_file = f"{model}_model_{str(i)}.joblib"
                    model_path = os.path.join(base_path, benchmark, model_file)

                    desc = f"{MODEL_TYPE}.{benchmark}.{vectorizer}.{model}.{i}"

                    if not os.path.exists(model_path):
                        print("skipped", desc)
                        continue
                    if os.path.exists(os.path.join(HYPERPARAMETER_SEARCH_PATH, f"{desc}.statistics.json")):
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

                    out_path = os.path.join(HYPERPARAMETER_SEARCH_PATH, f"{desc}.csv")
                    result_df.to_csv(out_path)

                    out_path = os.path.join(HYPERPARAMETER_SEARCH_PATH, f"{desc}.statistics.json")
                    with open(out_path, "w") as outfile:
                        json.dump(scores, outfile, indent=2)

                    print(time.time() - round_start)


def evaluate_hyperparameter_search_bert():
    base_path = BERT_MODEL_PATH
    models = ["biolinkbert", "pubmedbert"] if TASK == "RE" else ["pubmedbert", "biobert"]
    num_configs = 36
    lbp = BertPredictionPipeline()
    benchmarks = BENCHMARKS_RE if TASK == "RE" else BENCHMARKS_TC
    for benchmark in benchmarks:

        if TASK == "TC":
            gen = TextDataGeneration(benchmark, "test")
        elif TASK == "RE":
            if benchmark == "Chemprot_c":
                gen = ChemprotDataGeneration("test")
            else:
                gen = BaselineDataGeneration(benchmark, "test")
        else:
            raise NotImplementedError()

        for model in models:
            for i in range(0, num_configs):
                model_file = f"{model}_model_{str(i)}"
                model_path = os.path.join(base_path, benchmark)

                desc = f"{MODEL_TYPE}.{benchmark}.{model}.{i}"

                if not os.path.exists(os.path.join(model_path, model_file)):
                    print("skipped", desc)
                    continue

                if os.path.exists(os.path.join(HYPERPARAMETER_SEARCH_PATH, f"{desc}.statistics.json")):
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

                out_path = os.path.join(HYPERPARAMETER_SEARCH_PATH, f"{desc}.csv")
                result_df.to_csv(out_path)

                out_path = os.path.join(HYPERPARAMETER_SEARCH_PATH, f"{desc}.statistics.json")
                with open(out_path, "w") as outfile:
                    json.dump(scores, outfile, indent=2)

                print(time.time() - round_start)


if __name__ == "__main__":
    evaluate_hyperparameter_search_traditional()
    # evaluate_hyperparameter_search_bert()

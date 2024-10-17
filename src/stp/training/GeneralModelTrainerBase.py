import json
import os
import itertools
from abc import ABC, abstractmethod

from stp.benchmark import BENCHMARK_TO_NAME
from stp.data_generation.text.text_generation import TEXT_BENCHMARKS
from stp.run_config import TASK, MODEL_TYPE, BERT_CPU, TC_REDUCTION_VALUES, TC_FLIPPING_VALUES

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['VECLIB_MAXIMUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['OMP_THREAD_LIMIT'] = '32'


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes} minutes {seconds:.2f} seconds"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{hours} hours {minutes} minutes {seconds:.2f} seconds"


def load_best_parameters(model_type, method, data_source):
    best_parameters_map = {
        "traditional": {
            "tfidf": {
                "CDR": {
                    # "C": 1,
                    # "kernel": "rbf"
                    "colsample_bytree": 1.0,
                    "learning_rate": 0.01,
                    "max_depth": 3,
                    "n_estimators": 50,
                    "subsample": 1.0
                },
                "Chemprot": {
                    # "C": 0.1,
                    # "degree": 1,
                    # "kernel": "poly"
                    "colsample_bytree": 1.0,
                    "learning_rate": 0.01,
                    "max_depth": 5,
                    "n_estimators": 100,
                    "subsample": 0.8
                },
                "Chemprot_c": {
                    # "C": 1,
                    # "kernel": "sigmoid"
                    "colsample_bytree": 1.0,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "n_estimators": 100,
                    "subsample": 0.8
                },
                "DDI": {
                    # "C": 10,
                    # "degree": 2,
                    # "kernel": "poly"
                    "colsample_bytree": 1.0,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "n_estimators": 100,
                    "subsample": 1.0
                },
                "HallmarksOfCancer": {
                    # "C": 1,
                    # "kernel": "sigmoid"
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "n_estimators": 100,
                    "subsample": 0.8
                },
                "LongCovid": {
                    # "C": 1,
                    # "kernel": "sigmoid"
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "n_estimators": 50,
                    "subsample": 0.8
                },
                "Ohsumed": {
                    # "C": 1,
                    # "degree": 1,
                    # "kernel": "poly"
                    "colsample_bytree": 1.0,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "n_estimators": 100,
                    "subsample": 0.8
                },
                "PharmaTech": {
                    # "C": 1,
                    # "degree": 1,
                    # "kernel": "poly"
                    "colsample_bytree": 1.0,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "n_estimators": 50,
                    "subsample": 1.0
                }
            },
            "sbert": {
                "CDR": {
                    "C": 1,
                    "kernel": "rbf"
                },
                "Chemprot": {
                    "C": 0.1,
                    "degree": 1,
                    "kernel": "poly"
                },
                "Chemprot_c": {
                    "C": 1,
                    "kernel": "sigmoid"
                },
                "DDI": {
                    "C": 10,
                    "degree": 2,
                    "kernel": "poly"
                },
                "HallmarksOfCancer": {
                    "C": 1,
                    "kernel": "sigmoid"
                },
                "LongCovid": {
                    "C": 1,
                    "kernel": "sigmoid"
                },
                "Ohsumed": {
                    "C": 1,
                    "degree": 1,
                    "kernel": "poly"
                },
                "PharmaTech": {
                    "C": 1,
                    "degree": 1,
                    "kernel": "poly"
                }
            }
        },
        "bert": {
            "michiyasunaga/BioLinkBERT-base": {
                "CDR": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.1
                },
                "Chemprot": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 1,
                    "weight_decay": 0.3
                },
                "Chemprot_c": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 1,
                    "weight_decay": 0.3
                },
                "DDI": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 3,
                    "weight_decay": 0.3
                },
                "PharmaTech": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.1
                }
            },
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": {
                "CDR": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.3
                },
                "Chemprot": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 1,
                    "weight_decay": 0.1
                },
                "Chemprot_c": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 5,
                    "weight_decay": 0.0
                },
                "DDI": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 1,
                    "weight_decay": 0.3
                },
                "HallmarksOfCancer": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 5,
                    "weight_decay": 0.2
                },
                "LongCovid": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 3,
                    "weight_decay": 0.2
                },
                "Ohsumed": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 5,
                    "weight_decay": 0.3
                },
                "PharmaTech": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.0
                }
            },
            "dmis-lab/biobert-v1.1": {
                "HallmarksOfCancer": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 3,
                    "weight_decay": 0.3
                },
                "LongCovid": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 3,
                    "weight_decay": 0.3
                },
                "Ohsumed": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 5,
                    "weight_decay": 0.3
                },
                "PharmaTech": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.0
                }
            },
            "bert-base-uncased": {
                "PharmaTech": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.0
                }
            },
            "roberta-base": {
                "PharmaTech": {
                    "learning_rate": 1e-05,
                    "num_train_epochs": 3,
                    "weight_decay": 0.3
                }
            },
            "xlnet-base-cased": {
                "PharmaTech": {
                    "learning_rate": 0.0001,
                    "num_train_epochs": 1,
                    "weight_decay": 0.2
                }
            }
        }
    }

    return best_parameters_map[model_type][method][data_source]


class GeneralModelTrainerBase(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    def train_multi_class_model(self):
        print(f"Training {self.model_name} multi-class model")
        self.train_model_wrapper(None, None, "multi")

    def train_model_wrapper(self, df1_name, df2_name, data_type, reduction=None, flip=None):
        raise NotImplementedError("This method should be implemented.")

    def train_benchmark_model(self):
        print(f"Training {self.model_name} benchmark model")
        if TASK == "RE":
            for df_name in BENCHMARK_TO_NAME: #["DDI"]: #
                self.train_model_wrapper(df_name, None, "benchmark")
        elif TASK == "TC":
            for df_name in TEXT_BENCHMARKS:
                if MODEL_TYPE == "reduction":
                    for reduction in TC_REDUCTION_VALUES:
                        self.train_model_wrapper(df_name, None, "benchmark", reduction=reduction)
                elif MODEL_TYPE == "flipping":
                    for flip in TC_FLIPPING_VALUES:
                        self.train_model_wrapper(df_name, None, "benchmark", flip=flip)
                else:
                    self.train_model_wrapper(df_name, None, "benchmark")

    def train_chemprot_c_model(self):
        self.train_model_wrapper("Chemprot_c", None, "benchmark")

    def train_all_methods(self):
        # self.train_multi_class_model()
        self.train_benchmark_model()
        if MODEL_TYPE != "dsv" and TASK != 'TC' and not BERT_CPU:
            self.train_chemprot_c_model()

    def save_performance(self, model_folder, best_params, best_score, worst_params, worst_score):
        performance = {
            "best_params": best_params,
            "best_acc": best_score,
            "worst_params": worst_params,
            "worst_acc": worst_score
        }
        performance_path = os.path.join(model_folder, f"{self.model_name}_performance.txt")
        with open(performance_path, "w") as f:
            json.dump(performance, f, indent=4)

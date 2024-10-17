import os.path

from stp.benchmark import Benchmark
from stp.config import SBERT_DSV_MODELS, TFIDF_DSV_MODELS, SBERT_BASELINE_MODELS, \
    TFIDF_BASELINE_MODELS, BASELINE_BERT_MODELS, DSV_BERT_MODELS, GPT_BERT_MODELS, SBERT_GPT_MODELS, \
    TFIDF_GPT_MODELS, SBERT_TEXT_MODELS, TFIDF_TEXT_MODELS, TEXT_BERT_MODELS, MULTI_MODELS_TFIDF, MULTI_MODELS_SBERT, \
    MULTI_BERT_MODELS, LLAMA_BERT_MODELS, TFIDF_LLAMA_MODELS, SBERT_LLAMA_MODELS, MULTI_MODELS_TEXT_SBERT, \
    MULTI_MODELS_TEXT_TFIDF, MULTI_TEXT_BERT_MODELS, FLIPPING_MODELS_SBERT, FLIPPING_MODELS_TFIDF, FLIPPING_BERT_MODELS, \
    REDUCTION_MODELS_TFIDF, REDUCTION_MODELS_SBERT, REDUCTION_BERT_MODELS, HUGGINGFACE_TOKEN_PATH, OPENAI_TOKEN_PATH

USE_MEMORY_CACHE = True

# model training parameters
TASK = "TC"  # "TC" "RE"
MODEL_TYPE = "baseline"  # "dsv" "multi" "baseline" "llama" "gpt" "reduction" "flipping"
RANDOM_STATE = 42
RETRAIN_MODELS = False
CLASS_SIZE = 10000
PARAM_SEARCH = False
BALANCE_DATASET = True
TRAIN_DEFAULT_MODELS = False
BERT_CPU = False
SAVE_ALL_MODELS = False

# Distant super vision data preparation parameters
PUBMED_TOP_K_DOCUMENTS = 100
OBFUSCATE_ENTITY_TAGS = True

NUMBER_OF_PUBMED_DOCUMENTS = 36_555_430
AVG_SENTENCES_PER_DOCUMENT = 9.74

BENCHMARKS = {
    "CDR": Benchmark("CDR"),
    "Chemprot": Benchmark("Chemprot"),
    "DDI": Benchmark("DDI")
}

MODEL_TYPE_MAPPING = {
    "baseline": {
        "tfidf": TFIDF_BASELINE_MODELS if TASK == "RE" else TFIDF_TEXT_MODELS,
        "sbert": SBERT_BASELINE_MODELS if TASK == "RE" else SBERT_TEXT_MODELS,
    },
    "dsv": {
        "tfidf": TFIDF_DSV_MODELS,
        "sbert": SBERT_DSV_MODELS
    },
    "gpt": {
        "tfidf": TFIDF_GPT_MODELS,
        "sbert": SBERT_GPT_MODELS
    },
    "multi": {
        "tfidf": MULTI_MODELS_TFIDF if TASK == "RE" else MULTI_MODELS_TEXT_TFIDF,
        "sbert": MULTI_MODELS_SBERT if TASK == "RE" else MULTI_MODELS_TEXT_SBERT
    },
    "llama": {
        "tfidf": TFIDF_LLAMA_MODELS,
        "sbert": SBERT_LLAMA_MODELS
    },
    "reduction": {
        "tfidf": REDUCTION_MODELS_TFIDF,
        "sbert": REDUCTION_MODELS_SBERT
    },
    "flipping": {
        "tfidf": FLIPPING_MODELS_TFIDF,
        "sbert": FLIPPING_MODELS_SBERT
    }
}

BERT_MODEL_PATH_MAPPING = {
    "baseline": BASELINE_BERT_MODELS if TASK == "RE" else TEXT_BERT_MODELS,
    "dsv": DSV_BERT_MODELS,
    "gpt": GPT_BERT_MODELS,
    "multi": MULTI_BERT_MODELS if TASK == "RE" else MULTI_TEXT_BERT_MODELS,
    "llama": LLAMA_BERT_MODELS,
    "reduction": REDUCTION_BERT_MODELS,
    "flipping": FLIPPING_BERT_MODELS
}

BERT_MODEL_PATH = BERT_MODEL_PATH_MAPPING.get(MODEL_TYPE)

VECTORIZER = MODEL_TYPE_MAPPING.get(MODEL_TYPE)

CLASSIFIER = [
    "svc",
    "xgb",
    "rf",
]

BERT_MODEL_MAPPING = {
    # "bert": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    # "roberta": "roberta-base",
    # "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    # "xlnet": "xlnet-base-cased"
}

PARAM_GRIDS = {
    "SVC": [
        # {
        #     'C': [0.1, 1, 10],
        #     'gamma': [0.1, 10],
        #     'kernel': ['poly'],
        #     'degree': [1, 2]
        # },
        {
            "C": [0.1, 1, 10, 100],
            "kernel": ['poly'],
            "degree": [1, 2, 3, 4, 5, 6]
        },
        {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'sigmoid']
        }
    ],
    "XGBClassifier": [{
        # 'n_estimators': [300, 400],
        # 'max_depth': [5, 7],
        # 'colsample_bytree': [0.8, 1.0]
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }],
    "RandomForestClassifier": [{
        # 'n_estimators': [400, 500],
        # 'max_depth': [30, 40],
        # 'min_samples_split': [2, 5]
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }]
}

MODEL_VARIANT = {
    "model": "tuned_params",
    "model_worst": "worst_params",
    "default_model": "default_params"
}

MODEL_VARIANT_TO_PRINT_NAME = {
    "tuned_params": "Best parameters",
    "worst_params": "Worst parameters",
    "default_params": "Default parameters"
}

METHOD_TO_PRINT_NAME = {
    "binaryclass": "BinaryClass",
    "multiclass": "Multiclass",
    "one_vs_one": "One VS One",
    "one_vs_rest": "One VS Rest"
}

TC_REDUCTION_VALUES = [0.25, 0.5, 0.75]
TC_FLIPPING_VALUES = [0.25, 0.5, 0.75]


def load_api_token(api_token_path: str):
    if not os.path.exists(api_token_path) or not os.path.isfile(api_token_path):
        print("API token NOT found. File {} not found.".format(api_token_path))
        return ""

    with open(api_token_path, "rt") as f:
        print("API token found")
        return f.read().strip()


HUGGINGFACE_TOKEN = load_api_token(HUGGINGFACE_TOKEN_PATH)
OPENAI_TOKEN = load_api_token(OPENAI_TOKEN_PATH)

print("==" * 60)
print("Use memory cache       :", USE_MEMORY_CACHE)
print("DSV".center(120, '='))
print("Obfuscate entity tags  :", OBFUSCATE_ENTITY_TAGS)

print("Model Training".center(120, '='))
print("Random State           :", RANDOM_STATE)
print("Retrain Models         :", RETRAIN_MODELS)
print("Class Size (Train)     :", CLASS_SIZE)
print("Parameter Search       :", PARAM_SEARCH)
print("==" * 60)

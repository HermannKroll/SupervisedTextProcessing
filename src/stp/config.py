import os

GIT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
DATA_DIR = os.path.join(GIT_ROOT_DIR, "data")

RESOURCES_DIR = os.path.join(DATA_DIR, 'resources')

# benchmark stuff
BENCHMARKS_PATH = os.path.join(RESOURCES_DIR, 'benchmarks')
CDR_PATH = os.path.join(BENCHMARKS_PATH, 'CDR')
CHEMPROT_PATH = os.path.join(BENCHMARKS_PATH, 'ChemProt')
DDI_PATH = os.path.join(BENCHMARKS_PATH, 'DDI')

HALLMARKS_OF_CANCER_PATH = os.path.join(BENCHMARKS_PATH, 'HallmarksOfCancer')
LONG_COVID_PATH = os.path.join(BENCHMARKS_PATH, 'LongCovid')
OHSUMED_PATH = os.path.join(BENCHMARKS_PATH, 'Ohsumed')
PHARMA_TECH_PATH = os.path.join(BENCHMARKS_PATH, 'PharmaTech')

OPENAI_PATH = os.path.join(RESOURCES_DIR, "openai")
OPENAI_INPUT_PATH = os.path.join(OPENAI_PATH, "input")
OPENAI_OUTPUT_PATH = os.path.join(OPENAI_PATH, "output")
OPENAI_METADATA_PATH = os.path.join(OPENAI_PATH, "metadata")

# evaluation stuff
EVALUATION_PATH = os.path.join(DATA_DIR, 'evaluation')
BENCHMARK_RESULTS_PATH = os.path.join(EVALUATION_PATH, 'benchmark_results')
HYPERPARAMETER_SEARCH_PATH = os.path.join(EVALUATION_PATH, 'hyperparameter_search')
TC_NOICE_EVAL_PATH = os.path.join(EVALUATION_PATH, "tc_noise")
DIAGRAMS_PATH = os.path.join(EVALUATION_PATH, 'diagrams')
CONFUSION_MATRICES_PATH = os.path.join(DIAGRAMS_PATH, 'confusion_matrices')
BARPLOT_PATH = os.path.join(DIAGRAMS_PATH, 'barplot')
BOXPLOT_PATH = os.path.join(DIAGRAMS_PATH, 'boxplot')
TC_NOISE_PATH = os.path.join(DIAGRAMS_PATH, 'tc_noise')
CLASS_DISTRIBUTION_PATH = os.path.join(DIAGRAMS_PATH, 'class_distribution')
PROMPTING_PATH = os.path.join(EVALUATION_PATH, 'prompting')

# distant supervision knowledge-bases
DSV_KB_PATH = os.path.join(RESOURCES_DIR, 'knowledgebases')
CDR_DSV_KB_FILE = os.path.join(DSV_KB_PATH, 'CTD_chemicals_diseases.csv')
CHEMPROT_DSV_KB_FILE = os.path.join(DSV_KB_PATH, 'CTD_chem_gene_ixns.csv')
DDI_DSV_KB_FILE = os.path.join(DSV_KB_PATH, 'drugbank_database_2024.xml')

VOCAB_PATH = os.path.join(RESOURCES_DIR, 'vocabularies')
CHEMICALS_VOCAB_FILE = os.path.join(VOCAB_PATH, 'CTD_chemicals.csv')
DISEASES_VOCAB_FILE = os.path.join(VOCAB_PATH, 'CTD_diseases.csv')
GENES_VOCAB_FILE = os.path.join(VOCAB_PATH, 'CTD_genes.csv')
DRUGBANK_VOCAB_FILE = os.path.join(VOCAB_PATH, 'drugbank_vocabulary_2024.csv')

# cache
LM_CACHE_DIR = os.path.join(DATA_DIR, "lm_cache")

SENTENCES_PATH = os.path.join(DATA_DIR, "sentences")
LABELED_SENTENCES_PATH = os.path.join(SENTENCES_PATH, "labeled_sentences")
OLMO_SENTENCES_PATH = os.path.join(SENTENCES_PATH, "olmo_relabel")
LLAMA_SENTENCES_PATH = os.path.join(SENTENCES_PATH, "llama_relabel")
BIOMISTRAL_SENTENCES_PATH = os.path.join(SENTENCES_PATH, "biomistral_relabel")
OPENAI_SENTENCES_PATH = os.path.join(SENTENCES_PATH, "openai_relabel")
DSV_SENTENCES_PATH = os.path.join(SENTENCES_PATH, "dsv_relabel")

MODEL_PATH = os.path.join(DATA_DIR, "models")

TRADITIONAL_MODELS = os.path.join(MODEL_PATH, "traditional_models")
DSV_MODELS = os.path.join(TRADITIONAL_MODELS, "dsv")
SBERT_DSV_MODELS = os.path.join(DSV_MODELS, "sbert")
TFIDF_DSV_MODELS = os.path.join(DSV_MODELS, "tfidf")
BASELINE_MODELS = os.path.join(TRADITIONAL_MODELS, "baseline")
SBERT_BASELINE_MODELS = os.path.join(BASELINE_MODELS, "sbert")
TFIDF_BASELINE_MODELS = os.path.join(BASELINE_MODELS, "tfidf")
GPT_MODELS = os.path.join(TRADITIONAL_MODELS, "gpt")
SBERT_GPT_MODELS = os.path.join(GPT_MODELS, "sbert")
TFIDF_GPT_MODELS = os.path.join(GPT_MODELS, "tfidf")
LLAMA_MODELS = os.path.join(TRADITIONAL_MODELS, "llama")
SBERT_LLAMA_MODELS = os.path.join(LLAMA_MODELS, "sbert")
TFIDF_LLAMA_MODELS = os.path.join(LLAMA_MODELS, "tfidf")
TEXT_MODELS = os.path.join(TRADITIONAL_MODELS, "text")
SBERT_TEXT_MODELS = os.path.join(TEXT_MODELS, "sbert")
TFIDF_TEXT_MODELS = os.path.join(TEXT_MODELS, "tfidf")
MULTI_MODELS = os.path.join(TRADITIONAL_MODELS, "multitask")
MULTI_MODELS_SBERT = os.path.join(MULTI_MODELS, "sbert")
MULTI_MODELS_TFIDF = os.path.join(MULTI_MODELS, "tfidf")
MULTI_MODELS_TEXT = os.path.join(TRADITIONAL_MODELS, "multitask_text")
MULTI_MODELS_TEXT_SBERT = os.path.join(MULTI_MODELS_TEXT, "sbert")
MULTI_MODELS_TEXT_TFIDF = os.path.join(MULTI_MODELS_TEXT, "tfidf")
REDUCTION_MODELS = os.path.join(TRADITIONAL_MODELS, "reduction")
REDUCTION_MODELS_SBERT = os.path.join(REDUCTION_MODELS, "sbert")
REDUCTION_MODELS_TFIDF = os.path.join(REDUCTION_MODELS, "tfidf")
FLIPPING_MODELS = os.path.join(TRADITIONAL_MODELS, "flipping")
FLIPPING_MODELS_SBERT = os.path.join(FLIPPING_MODELS, "sbert")
FLIPPING_MODELS_TFIDF = os.path.join(FLIPPING_MODELS, "tfidf")

BERT_MODELS = os.path.join(MODEL_PATH, "bert_models")
BASELINE_BERT_MODELS = os.path.join(BERT_MODELS, "baseline")
DSV_BERT_MODELS = os.path.join(BERT_MODELS, "dsv")
GPT_BERT_MODELS = os.path.join(BERT_MODELS, "gpt")
LLAMA_BERT_MODELS = os.path.join(BERT_MODELS, "llama")
TEXT_BERT_MODELS = os.path.join(BERT_MODELS, "text")
MULTI_BERT_MODELS = os.path.join(BERT_MODELS, "multitask")
MULTI_TEXT_BERT_MODELS = os.path.join(BERT_MODELS, "multitask_text")
REDUCTION_BERT_MODELS = os.path.join(BERT_MODELS, "reduction")
FLIPPING_BERT_MODELS = os.path.join(BERT_MODELS, "flipping")

TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")

HUGGINGFACE_TOKEN_PATH = os.path.join(GIT_ROOT_DIR, "HUGGINGFACE_TOKEN")
OPENAI_TOKEN_PATH = os.path.join(GIT_ROOT_DIR, "OPENAI_TOKEN")

paths = [LM_CACHE_DIR, SENTENCES_PATH, LABELED_SENTENCES_PATH, MODEL_PATH, BERT_MODELS, BENCHMARK_RESULTS_PATH,
         BENCHMARK_RESULTS_PATH, DIAGRAMS_PATH, TRAINING_DATA_DIR, BASELINE_MODELS, SBERT_BASELINE_MODELS,
         TFIDF_BASELINE_MODELS, DSV_MODELS, SBERT_DSV_MODELS, TFIDF_DSV_MODELS, BASELINE_BERT_MODELS, DSV_BERT_MODELS,
         TRADITIONAL_MODELS, CONFUSION_MATRICES_PATH, OLMO_SENTENCES_PATH, GPT_MODELS, SBERT_GPT_MODELS,
         TFIDF_GPT_MODELS, GPT_BERT_MODELS, CONFUSION_MATRICES_PATH, OPENAI_PATH, OPENAI_INPUT_PATH, OPENAI_OUTPUT_PATH,
         OPENAI_METADATA_PATH, TEXT_BERT_MODELS, TEXT_MODELS, SBERT_TEXT_MODELS, TFIDF_TEXT_MODELS, MULTI_MODELS,
         MULTI_MODELS_SBERT, MULTI_MODELS_TFIDF, MULTI_BERT_MODELS, LLAMA_BERT_MODELS, LLAMA_MODELS, SBERT_LLAMA_MODELS,
         TFIDF_LLAMA_MODELS, BARPLOT_PATH, MULTI_MODELS_TEXT, MULTI_MODELS_TEXT_SBERT, MULTI_MODELS_TEXT_TFIDF,
         MULTI_TEXT_BERT_MODELS, REDUCTION_BERT_MODELS, REDUCTION_MODELS, REDUCTION_MODELS_SBERT,
         REDUCTION_MODELS_TFIDF, FLIPPING_MODELS_SBERT, FLIPPING_MODELS_TFIDF, FLIPPING_MODELS, FLIPPING_BERT_MODELS]

os.umask(0)  # to fix the 777 issue
for path in paths:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True, mode=0o777)

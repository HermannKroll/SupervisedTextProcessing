import os
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from stp.benchmark import BenchmarkLabel, BENCHMARK_TO_NAME, ChemprotLabel, MultitaskLabel
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.chemprot_generation import ChemprotDataGeneration
from stp.data_generation.dsv.dsv_generation import DSVDataGeneration
from stp.data_generation.text.text_generation import TextDataGeneration, TEXT_BENCHMARKS, transform_text_label
from stp.run_config import RANDOM_STATE, CLASS_SIZE, MODEL_TYPE, BALANCE_DATASET, TASK
from stp.config import TRAINING_DATA_DIR, LLAMA_SENTENCES_PATH, OPENAI_SENTENCES_PATH
from stp.util.dataframe import reduce_dataframe


def balance_dataset(df, min_class_size=None):
    if not BALANCE_DATASET:
        return df
    class_counts = df['label'].value_counts()
    class_counts_min = class_counts.min()
    # if MODEL_TYPE == "baseline" or MODEL_TYPE == "multi" or MODEL_TYPE == "gpt" or MODEL_TYPE == "llama":
    if not min_class_size:
        min_class_size = class_counts_min
    # elif MODEL_TYPE == "dsv":
    #     if CLASS_SIZE and not min_class_size:
    #         min_class_size = CLASS_SIZE
    #     elif not CLASS_SIZE and not min_class_size:
    #         min_class_size = class_counts_min
    #     if class_counts_min < min_class_size:
    #         min_class_size = class_counts_min
    balanced_dfs = []
    for label, count in class_counts.items():
        class_df = df[df['label'] == label]
        if count > min_class_size:
            balanced_class_df = reduce_dataframe(class_df, min_class_size)
            balanced_dfs.append(balanced_class_df)
        else:
            balanced_dfs.append(class_df)
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_df


def load_data_generator(df_name, split_DDI=True):
    if TASK == "RE":
        if MODEL_TYPE == "baseline" or MODEL_TYPE == "multi":
            if df_name == "DDI":
                if split_DDI:
                    gen = BaselineDataGeneration(data_source=df_name, run="train")
                    gen_train, gen_dev = gen.split_data(random_state=RANDOM_STATE)
                else:
                    gen_train = BaselineDataGeneration(data_source=df_name, run="train")
                    gen_dev = None
            elif df_name == "Chemprot_c":
                gen_train = ChemprotDataGeneration(run="train")
                gen_dev = ChemprotDataGeneration(run="dev")
            else:
                gen_train = BaselineDataGeneration(data_source=df_name, run="train")
                gen_dev = BaselineDataGeneration(data_source=df_name, run="dev")
            return gen_train, gen_dev
        elif MODEL_TYPE == "dsv":
            gen = DSVDataGeneration(data_source=df_name, run="train")
            return gen, None
    elif TASK == "TC":
        gen_train = TextDataGeneration(data_source=df_name, run="train")
        gen_dev = TextDataGeneration(data_source=df_name, run="dev")
        return gen_train, gen_dev

    # elif MODEL_TYPE == "dsv":
    #     gen = DSVDataGeneration(data_source=df_name)
    #     return gen, None

def filter_and_replace_labels(df, data_source, df_name):
    filtered_df = df[df["label"] == BenchmarkLabel.to_value(data_source)]
    if df_name == "DDI":
        filtered_df = filtered_df.copy()
        filtered_df.loc[:, 'label'] = filtered_df['label'].replace(
            BenchmarkLabel.to_value(data_source),
            MultitaskLabel.to_value(data_source)
        )
    return filtered_df

def concatenate_and_reduce(dfs, class_counts_min):
    count_non_empty_dfs = sum(len(df) > 0 for df in dfs)
    reduced_dfs = [reduce_dataframe(df, int(class_counts_min / count_non_empty_dfs)) for df in dfs]
    return pd.concat(reduced_dfs, ignore_index=True)


def prepare_multi_class_data():
    all_data = []
    all_dev_data = []

    negative_dfs = []
    negative_dev_dfs = []

    if TASK == "RE":
        for df_name in BENCHMARK_TO_NAME.keys():
            if df_name == "Chemprot":
                continue
            gen_train, gen_dev = load_data_generator(df_name)
            df = filter_and_replace_labels(gen_train.sentences, gen_train.data_source, df_name)
            if df_name == "DDI":
                df['label'] = df['label'].replace(BenchmarkLabel.to_value(gen_train.data_source), MultitaskLabel.to_value(gen_train.data_source))
            all_data.append(df)

            negative_dfs.append(gen_train.sentences[gen_train.sentences["label"] == BenchmarkLabel.NEGATIVE])
            if gen_dev:
                df_dev = filter_and_replace_labels(gen_dev.sentences, gen_dev.data_source, df_name)
                if df_name == "DDI":
                    df_dev['label'] = df_dev['label'].replace(BenchmarkLabel.to_value(gen_train.data_source),
                                                      MultitaskLabel.to_value(gen_train.data_source))
                all_dev_data.append(df_dev)
                negative_dev_dfs.append(gen_dev.sentences[gen_dev.sentences["label"] == BenchmarkLabel.NEGATIVE])

        chem_train, chem_dev = load_data_generator("Chemprot_c")
        chem_df = chem_train.sentences[chem_train.sentences["label"] != ChemprotLabel.NEGATIVE]
        all_data.append(chem_df)
        negative_dfs.append(chem_train.sentences[chem_train.sentences["label"] == ChemprotLabel.NEGATIVE])
        if chem_dev:
            chem_df_dev = chem_dev.sentences[chem_dev.sentences["label"] != ChemprotLabel.NEGATIVE]
            all_dev_data.append(chem_df_dev)
            negative_dev_dfs.append(chem_dev.sentences[chem_dev.sentences["label"] == ChemprotLabel.NEGATIVE])
    elif TASK == "TC":
        for df_name in TEXT_BENCHMARKS:
            print(df_name)
            gen_train, gen_dev = load_data_generator(df_name)
            negative_label = transform_text_label(df_name, 0)
            df = gen_train.sentences[gen_train.sentences["label"] != negative_label]
            negative_df = gen_train.sentences[gen_train.sentences["label"] == negative_label]
            negative_df.loc[:, "label"] = 0
            negative_dfs.append(negative_df)
            all_data.append(df)
            if gen_dev:
                df_dev = gen_train.sentences[gen_train.sentences["label"] != negative_label]
                negative_dev_df = gen_dev.sentences[gen_dev.sentences["label"] == negative_label]
                negative_dev_df.loc[:, "label"] = 0
                negative_dev_dfs.append(negative_df)
                all_dev_data.append(df_dev)

    positive_df = pd.concat(all_data, ignore_index=True)
    positive_df = balance_dataset(positive_df)
    class_counts_min = positive_df['label'].value_counts().min()
    if len(negative_dfs) > 0:
        negative_df = concatenate_and_reduce(negative_dfs, class_counts_min)
        df = pd.concat([positive_df, negative_df], ignore_index=True)
    else:
        df = positive_df

    if all_dev_data:
        positive_dev_df = pd.concat(all_dev_data, ignore_index=True)
        class_counts_min = positive_dev_df['label'].value_counts().min()
        if len(negative_dfs) > 0:
            negative_dev_df = concatenate_and_reduce(negative_dev_dfs, class_counts_min)
            dev_df = pd.concat([positive_dev_df, negative_dev_df], ignore_index=True)
        else:
            dev_df = positive_dev_df
    else:
        dev_df = None

    class_counts = df['label'].value_counts()
    print(f"multiclass train : {class_counts}")

    if dev_df is not None:
        class_counts_dev = dev_df['label'].value_counts()
        print(f"multiclass dev : {class_counts_dev}")

    return df, dev_df


def prepare_benchmark_data(benchmark, reduction=None, flip=None):
    if MODEL_TYPE == "llama":
        gen, dev_gen = None, None
        df = pd.read_csv(os.path.join(LLAMA_SENTENCES_PATH, f"{benchmark}.three_yes.relabeled.csv"))
    elif MODEL_TYPE == "gpt":
        gen, dev_gen = None, None
        df = pd.read_csv(os.path.join(OPENAI_SENTENCES_PATH, f"{benchmark}.two_yes.relabeled.csv"))
    else:
        gen, dev_gen = load_data_generator(benchmark)
        df = gen.sentences
    if reduction:
        df = reduce_dataframe(df, int(len(df) * reduction))
    df = balance_dataset(df)

    if flip:
        n_samples_to_flip = int(len(df) * flip)
        df_to_flip = df.sample(n=n_samples_to_flip, random_state=RANDOM_STATE)
        remaining_df = df.drop(df_to_flip.index)
        shuffled_labels = df_to_flip['label'].sample(frac=1, random_state=RANDOM_STATE).values
        df_to_flip.loc[:, 'label'] = shuffled_labels
        df = pd.concat([remaining_df, df_to_flip]).sort_index()

    if dev_gen:
        dev_df = dev_gen.sentences
        print(f"Train: X:{len(df['sentence']) if TASK == 'RE' else len(df['text'])}, y:{len(df['label'])}")
        print(f"Dev: X:{len(df['sentence']) if TASK == 'RE' else len(df['text'])}, y:{len(dev_df['label'])}")
    else:
        dev_df = None

    class_counts = df['label'].value_counts()
    print(f"{benchmark} train : {class_counts}")

    if dev_df is not None:
        class_counts_dev = dev_df['label'].value_counts()
        print(f"{benchmark} dev : {class_counts_dev}")

    return df, dev_df


def prepare_data(sentence_transformer=None, **kwargs):
    print("Preparing data for training.")

    data_type = kwargs.get("data_type")
    df1_name = kwargs.get("df1_name")
    reduction = kwargs.get("reduction")
    flip = kwargs.get("flip")

    if data_type == "multi":
        cache_filename = f"{MODEL_TYPE}_multiclass_df.pkl"
        dimensions_filename = f"{MODEL_TYPE}_multiclass_dimensions.txt"
    elif data_type == "benchmark":
        if reduction:
            cache_filename = f"{MODEL_TYPE}_{df1_name}_{int(reduction * 100)}_df.pkl"
            dimensions_filename = f"{MODEL_TYPE}_{df1_name}_{int(reduction * 100)}_dimensions.txt"
        elif flip:
            cache_filename = f"{MODEL_TYPE}_{df1_name}_{int(flip * 100)}_df.pkl"
            dimensions_filename = f"{MODEL_TYPE}_{df1_name}_{int(flip * 100)}_dimensions.txt"
        else:
            cache_filename = f"{MODEL_TYPE}_{df1_name}_df.pkl"
            dimensions_filename = f"{MODEL_TYPE}_{df1_name}_dimensions.txt"
    else:
        raise ValueError("data_type must be one of 'benchmark' or 'multi'")

    dimensions_path = os.path.join(TRAINING_DATA_DIR, dimensions_filename)
    cache_path = os.path.join(TRAINING_DATA_DIR, cache_filename)

    if os.path.exists(cache_path):
        print("Using cached DataFrame.")
        x_train, x_dev, y_train, y_dev, model, pca, label_encoder = joblib.load(cache_path)
    else:
        if data_type == "multi":
            df, dev_df = prepare_multi_class_data()
            method = "multiclass"
        elif data_type == "benchmark":
            if reduction:
                print(f"Reducing dataset to {int(reduction * 100)}%")
                df, dev_df = prepare_benchmark_data(df1_name, reduction=reduction)
            elif flip:
                print(f"Flipping {int(flip * 100)}% of data")
                df, dev_df = prepare_benchmark_data(df1_name, flip=flip)
            else:
                df, dev_df = prepare_benchmark_data(df1_name)
            method = df1_name
        else:
            raise ValueError("data_type must be one of 'benchmark' or 'multi'")

        print(f"Method: {method}, Vectorizer: {sentence_transformer}")
        print(df.keys())
        corpus = df["sentence"] if TASK == "RE" else df["text"]
        labels = df["label"]
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        if dev_df is not None:
            corpus_dev = dev_df["sentence"] if TASK == "RE" else dev_df["text"]
            labels_dev = dev_df["label"]
            labels_dev_encoded = label_encoder.transform(labels_dev)
        else:
            corpus_dev = None
            labels_dev = None
            labels_dev_encoded = None

        pca = None

        if sentence_transformer == "sbert":
            model = SentenceTransformer('all-MiniLM-L6-v2')
            corpus_embedding = model.encode(corpus)
            if corpus_dev is not None:
                corpus_dev_embedding = model.encode(corpus_dev)
        elif sentence_transformer == "tfidf":
            model = TfidfVectorizer(stop_words="english")
            model.fit(corpus)
            corpus_embedding = model.transform(corpus)
            original_dimensions = corpus_embedding.shape[1]
            if corpus_dev is not None:
                corpus_dev_embedding = model.transform(corpus_dev)
            else:
                corpus_dev_embedding = None
            pca = PCA()
            pca.fit(corpus_embedding.toarray())
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            d = np.argmax(cumsum >= 0.95) + 1
            pca = PCA(n_components=d)
            pca.fit(corpus_embedding.toarray())
            corpus_embedding = pca.transform(corpus_embedding.toarray())
            if corpus_dev is not None:
                corpus_dev_embedding = pca.transform(corpus_dev_embedding.toarray())

            with open(dimensions_path, "w") as file:
                file.write(f"Number of dimensions before PCA: {original_dimensions}\n")
                file.write(f"Number of dimensions after PCA: {d}\n")
        elif sentence_transformer == "count":
            model = CountVectorizer()
            corpus_embedding = model.fit_transform(corpus)
            if corpus_dev is not None:
                corpus_dev_embedding = model.transform(corpus_dev)
        else:
            raise NotImplementedError("Sentence Transformer method unknown:", sentence_transformer)

        if not os.path.exists(kwargs["path"]):
            os.makedirs(kwargs["path"])

        if MODEL_TYPE == "dsv" or MODEL_TYPE == "gpt" or MODEL_TYPE == "llama":
            x_train = corpus_embedding
            y_train = labels_encoded
            x_dev, y_dev = None, None
        elif MODEL_TYPE == "baseline" or MODEL_TYPE == "multi" or MODEL_TYPE == "reduction" or MODEL_TYPE == "flipping":
            x_train = corpus_embedding
            y_train = labels_encoded
            x_dev = corpus_dev_embedding if corpus_dev is not None else None
            y_dev = labels_dev_encoded if labels_dev is not None else None

        joblib.dump((x_train, x_dev, y_train, y_dev, model, pca, label_encoder), cache_path)

    return x_train, x_dev, y_train, y_dev, model, pca, label_encoder


def prepare_bert_data(sentence_transformer=None, **kwargs):
    print("Preparing data for training.")

    data_type = kwargs.get("data_type")
    df1_name = kwargs.get("df1_name")
    reduction = kwargs.get("reduction")
    flip = kwargs.get("flip")

    if data_type == "multi":
        cache_filename = f"bert_{MODEL_TYPE}_multiclass_df.pkl"
    elif data_type == "benchmark":
        if reduction:
            cache_filename = f"bert_{MODEL_TYPE}_{df1_name}_{int(reduction * 100)}_df.pkl"
        elif flip:
            cache_filename = f"bert_{MODEL_TYPE}_{df1_name}_{int(flip * 100)}_df.pkl"
        else:
            cache_filename = f"bert_{MODEL_TYPE}_{df1_name}_df.pkl"
    else:
        raise ValueError("data_type must be one of 'benchmark' or 'multi'")

    cache_path = os.path.join(TRAINING_DATA_DIR, cache_filename)

    if os.path.exists(cache_path):
        print("Using cached DataFrame.")
        corpus, labels_encoded, corpus_dev, labels_dev_encoded, label_encoder, num_labels = joblib.load(cache_path)
    else:
        if data_type == "multi":
            df, dev_df = prepare_multi_class_data()
            method = "multiclass"
        elif data_type == "benchmark":
            if reduction:
                print(f"Reducing dataset to {int(reduction * 100)}%")
                df, dev_df = prepare_benchmark_data(df1_name, reduction)
            elif flip:
                print(f"Flipping {int(flip * 100)}% of data")
                df, dev_df = prepare_benchmark_data(df1_name, flip=flip)
            else:
                df, dev_df = prepare_benchmark_data(df1_name)
            method = df1_name
        else:
            raise ValueError("data_type must be one of 'benchmark' or 'multi'")

        print(f"Method: {method}, Vectorizer: {sentence_transformer}")

        num_labels = df['label'].nunique()
        corpus = df["sentence"] if TASK == "RE" else df["text"]
        labels = df["label"]
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        if dev_df is not None:
            corpus_dev = dev_df["sentence"] if TASK == "RE" else dev_df["text"]
            labels_dev = dev_df["label"]

            labels_dev_encoded = label_encoder.transform(labels_dev)
        else:
            corpus_dev = None
            labels_dev_encoded = None
            if MODEL_TYPE == "dsv":
                # corpus, corpus_dev, labels_encoded, labels_dev_encoded = train_test_split(corpus, labels_encoded,
                #                                                                           test_size=0.25,
                #                                                                           random_state=RANDOM_STATE)
                corpus_dev, labels_dev_encoded, labels_dev_encoded = None, None, None

        joblib.dump((corpus, labels_encoded, corpus_dev, labels_dev_encoded, label_encoder, num_labels), cache_path)

    return corpus, labels_encoded, corpus_dev, labels_dev_encoded, label_encoder, num_labels


def main():
    prepare_multi_class_data()
    # prepare_benchmark_data("Chemprot")
    # prepare_benchmark_data("DDI")
    # prepare_benchmark_data("CDR")
    # prepare_benchmark_data("HallmarksOfCancer")
    # prepare_benchmark_data("LongCovid")
    # prepare_benchmark_data("Ohsumed")
    # prepare_benchmark_data("PharmaTech")


if __name__ == "__main__":
    main()

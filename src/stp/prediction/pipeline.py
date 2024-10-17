import os
import time
from typing import List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from stp.config import SBERT_BASELINE_MODELS
from stp.run_config import TASK


class PredictionPipeline:
    @staticmethod
    def load_model(model_path):
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{model_path}' not found.")

        components = joblib.load(model_path)

        vectorizer = None
        pca = None
        label_encoder = None
        model = None

        # assure the order
        for component in components:
            if isinstance(component, (TfidfVectorizer, CountVectorizer, SentenceTransformer)):
                vectorizer = component
            elif isinstance(component, (PCA, type(None))):
                pca = component
            elif isinstance(component, LabelEncoder):
                label_encoder = component
            elif isinstance(component, (XGBClassifier, SVC, RandomForestClassifier)):
                model = component
            else:
                raise ValueError(f"Unexpected component type: {type(component)}")

        if vectorizer is None or label_encoder is None or model is None:
            raise ValueError("Not all required components were found in the model file.")
        if pca is None and "tfidf" in model_path:
            raise ValueError("tfidf vectorizer requires PCA object.")

        return vectorizer, pca, label_encoder, model

    @staticmethod
    def vectorize_sentence(sentences: List[str], vectorizer, pca):
        if isinstance(vectorizer, SentenceTransformer):
            vectorized_sentence = vectorizer.encode(sentences)
        elif isinstance(vectorizer, TfidfVectorizer):
            vectorized_sentence = vectorizer.transform(sentences)
            vectorized_sentence = pca.transform(vectorized_sentence.toarray())
        elif isinstance(vectorizer, CountVectorizer):
            vectorized_sentence = vectorizer.transform(sentences)
        else:
            raise NotImplementedError(f"Vectorizer method unknown:", type(vectorizer))
        return vectorized_sentence

    def predict_label(self, sentences, model_path):
        vectorizer, pca, label_encoder, model = self.load_model(model_path)

        input_key = "sentence" if TASK == "RE" else "text"

        start = time.time()
        # vectorize input
        model_inputs = self.vectorize_sentence(sentences[input_key].tolist(), vectorizer, pca)

        # predict labels for input
        y_pred = model.predict(model_inputs)

        # decode
        y_true = label_encoder.transform(sentences["label"])

        predicted_labels = label_encoder.inverse_transform(y_pred)
        time_taken = time.time() - start

        # build statistics
        require_multiclass = any((b in model_path) for b in {"Chemprot_c", "multitask", "HallmarksOfCancer", "Ohsumed"})
        average = "macro" if require_multiclass else "binary"

        scores = dict(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average=average),
            recall=recall_score(y_true, y_pred, average=average),
            f1_score=f1_score(y_true, y_pred, average=average),
            time_taken=time_taken,
            sentences=len(sentences[input_key])
        )

        return predicted_labels, scores


def main():
    sentences = ["Pharmacologically controlled drinking in the treatment of alcohol dependence or alcohol use "
                 "disorders: a systematic review with direct and network meta-analyses on nalmefene, [Drug], [Drug], "
                 "baclofen and topiramate.",
                 "Pharmacologically controlled drinking in the treatment of alcohol dependence or alcohol use "
                 "disorders: a systematic review with direct and network meta-analyses on nalmefene, [Drug], [Drug], "
                 "baclofen and topiramate."
                 ]
    model_path = os.path.join(SBERT_BASELINE_MODELS, os.path.join("multiclass", "svc_model.joblib"))
    pipeline = PredictionPipeline()
    predicted_label = pipeline.predict_label(sentences, model_path)
    print("Predicted Label:", predicted_label)

    model_path = os.path.join(SBERT_BASELINE_MODELS, os.path.join("multiclass", "rf_model.joblib"))
    pipeline = PredictionPipeline()
    predicted_label, probabilities, *_ = pipeline.predict_label(sentences, model_path, with_probabilities=True)
    print("Predicted Label:", predicted_label)
    print("Probabilities:", probabilities)


if __name__ == "__main__":
    main()

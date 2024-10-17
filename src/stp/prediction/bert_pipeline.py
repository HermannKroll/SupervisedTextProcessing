import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import joblib
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stp.config import BASELINE_BERT_MODELS
from stp.run_config import TASK
from stp.training.bert.BertModelTrainerBase import encode_data

BATCH_SIZE_GPU = 256


class BertPredictionPipeline:
    def __init__(self):
        self.model_cache = dict()

    @staticmethod
    def load_model(model_folder, model_name, device):
        if not os.path.exists(model_folder):
            raise ValueError(f"Model '{model_folder}' not found.")

        model_path = os.path.join(model_folder, f"{model_name}_model")
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device)
        label_encoder_path = os.path.join(model_folder, f"{model_name.split('_')[0]}_label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        return tokenizer, label_encoder, model

    def predict_label(self, sentences, model_folder, model_name, device="cpu"):
        assert device in {"cpu", "cuda"}

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Cuda is not available.")

        tokenizer, label_encoder, model = self.load_model(model_folder, model_name, device)
        sentences_index = "sentence" if TASK == "RE" else "text"

        # chose 256 batches for GPU and full-sized batches for cpu
        if device == "cpu":
            batch_size = len(sentences[sentences_index])
        else:
            batch_size = BATCH_SIZE_GPU

        time_taken = 0.0
        y_pred = list()
        y_pred_labels = list()

        # set model to evaluate
        model.eval()
        # disable gradient calculation
        with torch.no_grad():

            sentences_list = sentences[sentences_index].tolist()
            for i in range(0, len(sentences), batch_size):
                start = time.time()

                # encode input and predict classes
                sentence_list = sentences_list[i: i + batch_size]
                encoded_inputs = encode_data(tokenizer, sentence_list, max_length=256, device=device)
                outputs = model(**encoded_inputs)

                # decode classes and labels from output tensor
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes_tensor = torch.argmax(probabilities, dim=1)
                predicted_classes = predicted_classes_tensor.cpu().numpy()
                predicted_labels = label_encoder.inverse_transform(predicted_classes)

                time_taken += time.time() - start

                y_pred.extend(predicted_classes)
                y_pred_labels.extend(predicted_labels)

        # build statistics
        y_true = label_encoder.transform(sentences["label"].tolist())
        multi_labels = {"Chemprot_c", "multitask", "Ohsumed", "HallmarksOfCancer"}
        average = "macro" if any((c in model_folder) for c in multi_labels) else "binary"

        scores = dict(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average=average),
            recall=recall_score(y_true, y_pred, average=average),
            f1_score=f1_score(y_true, y_pred, average=average),
            time_taken=time_taken,
            sentences=len(sentences[sentences_index])
        )
        return y_pred_labels, scores


def main():
    sentences = ["Pharmacologically controlled drinking in the treatment of alcohol dependence or alcohol use "
                 "disorders: a systematic review with direct and network meta-analyses on nalmefene, [Drug], [Drug], "
                 "baclofen and topiramate.",
                 "Pharmacologically controlled drinking in the treatment of alcohol dependence or alcohol use "
                 "disorders: a systematic review with direct and network meta-analyses on nalmefene, [Drug], [Drug], "
                 "baclofen and topiramate."
                 ]
    model_name = "biobert"
    model_folder = os.path.join(BASELINE_BERT_MODELS, "multiclass")
    pipeline = BertPredictionPipeline()
    predicted_label = pipeline.predict_label(sentences, model_folder, model_name)
    print("Predicted Label:", predicted_label)

    model_name = "bert"
    model_folder = os.path.join(BASELINE_BERT_MODELS, "multiclass")
    pipeline = BertPredictionPipeline()
    predicted_label, probabilities = pipeline.predict_label(sentences, model_folder, model_name)
    print("Predicted Label:", predicted_label)
    print("Probabilities:", probabilities)


if __name__ == "__main__":
    main()

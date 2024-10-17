import json
import os

import torch
from transformers import TrainerCallback, PreTrainedTokenizerFast

from stp.training.GeneralModelTrainerBase import GeneralModelTrainerBase, load_best_parameters
from stp.training.data_preparation import prepare_bert_data
from stp.run_config import BERT_MODEL_PATH, MODEL_TYPE, BERT_CPU, TC_REDUCTION_VALUES, TC_FLIPPING_VALUES
from stp.config import LM_CACHE_DIR

os.environ["HF_MODELS_CACHE"] = LM_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def encode_data(tokenizer: PreTrainedTokenizerFast, sentences, labels=None, max_length=256, device='cpu'):
    encoded = tokenizer.batch_encode_plus(
        sentences,
        truncation=True,
        padding='longest',
        max_length=max_length,
        return_tensors='pt'
    ).to(device)
    if labels is not None:
        encoded['labels'] = torch.tensor(labels, device=device)
    return encoded


class BertModelTrainerBase(GeneralModelTrainerBase):
    def __init__(self, model_name):
        super().__init__(model_name)

    def train_model(self, X_train, y_train, X_dev, y_dev, label_encoder, num_labels, model_folder, parameter_list=None):
        pass

    def train_model_wrapper(self, df1_name, df2_name, data_type, reduction=None, flip=None):
        if data_type == "benchmark":
            if BERT_CPU:
                model_folder = os.path.join(os.path.join(BERT_MODEL_PATH, "cpu"), f"{df1_name}")
            elif reduction:
                model_folder = os.path.join(BERT_MODEL_PATH, f"{df1_name}_{int(reduction * 100)}")
            elif flip:
                model_folder = os.path.join(BERT_MODEL_PATH, f"{df1_name}_{int(flip * 100)}")
            else:
                model_folder = os.path.join(BERT_MODEL_PATH, f"{df1_name}")
            X_train, y_train, X_dev, y_dev, label_encoder, num_labels = prepare_bert_data(df1_name=df1_name,
                                                                                              path=model_folder,
                                                                                              data_type=data_type,
                                                                                              reduction=reduction,
                                                                                              flip=flip)
        elif data_type:  # multi
            if BERT_CPU:
                model_folder = os.path.join(BERT_MODEL_PATH, "cpu")
            else:
                model_folder = BERT_MODEL_PATH
            X_train, y_train, X_dev, y_dev, label_encoder, num_labels = prepare_bert_data(path=model_folder,
                                                                                          data_type=data_type,
                                                                                          reduction=reduction,
                                                                                          flip=flip)

        if MODEL_TYPE != "baseline" and MODEL_TYPE != "multi":
            parameter_list = load_best_parameters("bert", self.model_name, df1_name)
        else:
            parameter_list = None
        self.train_model(X_train, y_train, X_dev, y_dev, label_encoder, num_labels, model_folder,
                         parameter_list=parameter_list)

    class SaveEvalResultsCallback(TrainerCallback):
        def __init__(self, save_path):
            self.save_path = save_path
            self.eval_results = []

            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    self.eval_results = json.load(f)

        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs['metrics']
            self.eval_results.append(metrics)
            with open(self.save_path, 'w') as f:
                json.dump(self.eval_results, f, indent=4)

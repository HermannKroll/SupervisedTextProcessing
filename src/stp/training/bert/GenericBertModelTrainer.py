import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def configure_cpu_usage():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['OPENBLAS_NUM_THREADS'] = '32'
    os.environ['MKL_NUM_THREADS'] = '32'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '32'
    os.environ['NUMEXPR_NUM_THREADS'] = '32'
    torch.set_num_threads(32)
    n_cpus = range(40, 73)

    pid = os.getpid()
    print("PID: %i" % pid)

    cpu_arg = ','.join([str(ci) for ci in n_cpus])
    cmd = 'taskset -cp %s %i' % (cpu_arg, pid)
    print("Executing command '%s' ..." % cmd)
    os.system(cmd)


import json
import shutil
import joblib
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import Dataset
import time
from sklearn.model_selection import ParameterGrid

from stp.run_config import RETRAIN_MODELS, BERT_MODEL_MAPPING, TRAIN_DEFAULT_MODELS, SAVE_ALL_MODELS
from stp.training.bert.BertModelTrainerBase import BertModelTrainerBase, encode_data
from stp.training.GeneralModelTrainerBase import format_time


class GenericBertModelTrainer(BertModelTrainerBase):
    def __init__(self, name, use_cpu=False):
        super().__init__(name)
        self.name = name
        self.model_name = BERT_MODEL_MAPPING[name]
        self.use_cpu = use_cpu
        if self.use_cpu:
            configure_cpu_usage()

    def train_model(self, X_train, y_train, X_dev, y_dev, label_encoder, num_labels, model_folder,
                    parameter_list=None):
        model_path = os.path.join(model_folder, f"{self.name}_model")
        default_model_path = os.path.join(model_folder, f"{self.name}_default_model")
        worst_model_path = os.path.join(model_folder, f"{self.name}_worst_model")
        logging_path = os.path.join(model_folder, f'{self.name}_model_logging.json')
        label_encoder_path = os.path.join(model_folder, f'{self.name}_label_encoder.pkl')
        runtime_path = os.path.join(model_folder, f'{self.name}_runtime.txt')
        best_params_path = os.path.join(model_folder, f'{self.name}_best_params.txt')

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        joblib.dump(label_encoder, label_encoder_path)

        if not os.path.exists(model_path) or RETRAIN_MODELS: #True:#
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(worst_model_path):
                os.makedirs(worst_model_path)

            if parameter_list:
                param_grid = [parameter_list]
            else:
                param_grid = ParameterGrid({
                    'learning_rate': [1e-3, 1e-4, 1e-5],
                    'num_train_epochs': [1, 3, 5],
                    'weight_decay': [0.0, 0.1, 0.2, 0.3]
                })

            # Read already trained hyperparameters from the runtime file
            trained_params = set()
            if os.path.exists(runtime_path):
                with open(runtime_path, 'r') as f:
                    for line in f:
                        if 'epochs' in line and 'learning rate' in line and 'weight decay' in line:
                            parts = line.split(',')
                            epochs = int(parts[0].split()[0])
                            lr = float(parts[1].split()[0])
                            wd = float(parts[2].split()[0])
                            trained_params.add((epochs, lr, wd))

            best_eval_loss = float('inf')
            best_params = None
            worst_eval_loss = float('-inf')
            total_time = 0

            with open(runtime_path, 'a') as f:
                f.write("Training times:\n\n")

            for params in param_grid:
                learning_rate = params['learning_rate']
                num_epochs = params['num_train_epochs']
                weight_decay = params['weight_decay']

                if (num_epochs, learning_rate, weight_decay) in trained_params:
                    print(
                        f"Skipping already trained combination: {num_epochs} epochs, {learning_rate} learning rate, {weight_decay} weight decay")
                    continue

                device = torch.device("cpu") if self.use_cpu else torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")

                print(device)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
                model.to(device)
                train_encoded = encode_data(tokenizer, X_train.tolist(), y_train, max_length=512)
                dev_encoded = encode_data(tokenizer, X_dev.tolist(), y_dev,
                                          max_length=512) if not parameter_list else {}

                train_dataset = Dataset.from_dict(train_encoded)
                dev_dataset = Dataset.from_dict(dev_encoded)

                training_args = TrainingArguments(
                    output_dir='./results',
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=32,
                    per_device_eval_batch_size=64,
                    weight_decay=weight_decay,
                    fp16=True, #not usable for cpu training
                    logging_dir='./logs',
                    evaluation_strategy="epoch" if not parameter_list else "no",
                    save_strategy="epoch" if not parameter_list else "no",
                    learning_rate=learning_rate,
                    load_best_model_at_end=True if not parameter_list else False,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    # eval_dataset=dev_dataset,
                    # callbacks=[self.SaveEvalResultsCallback(logging_path)]
                )

                print(
                    f"Begin fine-tuning of {self.model_name} with {num_epochs} epochs, {learning_rate} learning rate, and {weight_decay} weight decay")
                start_time = time.time()
                trainer.train()
                end_time = time.time()
                training_time = end_time - start_time
                total_time += training_time
                print(f"Finished fine-tuning of {self.model_name} in {format_time(training_time)}.")

                # with open(logging_path, 'r') as f:
                #     eval_results = json.load(f)
                #
                # if isinstance(eval_results, list) and eval_results:
                #     current_eval_loss = eval_results[-1]['eval_loss']
                # else:
                #     raise ValueError("Invalid format for eval_results")

                if SAVE_ALL_MODELS:
                    model_index_path = os.path.join(model_folder, f"{self.name}_model_{idx}")
                    if not os.path.exists(model_index_path):
                        os.makedirs(model_index_path)
                    trainer.save_model(model_index_path)
                    tokenizer.save_pretrained(model_index_path)
                    print(f"Saved model with index {idx} to {model_index_path}")

                if not parameter_list:
                    if current_eval_loss < best_eval_loss:
                        best_eval_loss = current_eval_loss
                        best_params = params
                        if os.path.exists(model_path):
                            shutil.rmtree(model_path)
                        trainer.save_model(model_path)
                        tokenizer.save_pretrained(model_path)
                        print(
                            f"Saved best model with {num_epochs} epochs, {learning_rate} learning rate, and {weight_decay} weight decay to {model_path}")

                    if current_eval_loss > worst_eval_loss:
                        worst_eval_loss = current_eval_loss
                        if os.path.exists(worst_model_path):
                            shutil.rmtree(worst_model_path)
                        trainer.save_model(worst_model_path)
                        tokenizer.save_pretrained(worst_model_path)
                        print(
                            f"Saved worst model with {num_epochs} epochs, {learning_rate} learning rate, and {weight_decay} weight decay to {worst_model_path}")

                    with open(runtime_path, 'a') as f:
                        f.write(
                            f"{num_epochs} epochs, {learning_rate} learning rate, {weight_decay} weight decay: {format_time(training_time)}\n")

                    with open(best_params_path, 'w') as f:
                        json.dump(best_params, f)

                    print(f"Best evaluation loss: {best_eval_loss}")
                    print(f"Worst evaluation loss: {worst_eval_loss}")
                else:
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    trainer.save_model(model_path)
                    tokenizer.save_pretrained(model_path)
                    print(
                        f"Saved model with {num_epochs} epochs, {learning_rate} learning rate, and {weight_decay} weight decay to {model_path}")
                    with open(runtime_path, 'w') as f:
                        f.write(f"training_time: {format_time(training_time)}\n")

                    break
            if not parameter_list:
                with open(runtime_path, 'a') as f:
                    f.write(f"Overall total training time: {format_time(total_time)}\n")

                    f.write(f"Best evaluation loss: {best_eval_loss}\n")

                    f.write(f"Worst evaluation loss: {worst_eval_loss}\n")

            print(f"Total training time: {format_time(total_time)}")
        if TRAIN_DEFAULT_MODELS:
            print("Training BERT model with default parameters.")
            device = torch.device("cpu") if self.use_cpu else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
            model.to(device)
            train_encoded = encode_data(tokenizer, X_train.tolist(), y_train, max_length=512)
            train_dataset = Dataset.from_dict(train_encoded)
            training_args = TrainingArguments(
                output_dir='./results',
                logging_dir='./logs',
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )
            print(f"Begin training of {self.model_name} with default parameters.")
            start_time_default = time.time()
            trainer.train()
            end_time_default = time.time()
            training_time_default = end_time_default - start_time_default
            print(
                f"Finished training of {self.model_name} with default parameters in {format_time(training_time_default)}.")
            if os.path.exists(default_model_path):
                shutil.rmtree(default_model_path)
            trainer.save_model(default_model_path)
            tokenizer.save_pretrained(default_model_path)
            print(f"Saved default model to {default_model_path}")
            with open(runtime_path, 'a') as f:
                f.write(f"Training time for default model: {format_time(training_time_default)}\n")


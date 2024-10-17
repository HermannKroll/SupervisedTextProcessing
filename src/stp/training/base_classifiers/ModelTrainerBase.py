import os

from sklearn.metrics import accuracy_score

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_THREAD_LIMIT'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['VECLIB_MAXIMUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['JOBLIB_MULTIPROCESSING'] = '1'
import json
from time import time

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from concurrent.futures import ThreadPoolExecutor, as_completed

from stp.training.GeneralModelTrainerBase import GeneralModelTrainerBase, format_time, load_best_parameters
from stp.training.data_preparation import prepare_data
from stp.run_config import VECTORIZER, MODEL_TYPE


def train_and_evaluate(model, params, X_train, y_train, X_dev, y_dev):
    """
    Train a model with given parameters and evaluate it on the development set.

    Args:
    - model: The model to be trained.
    - params: The parameters for the model.
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_dev: Development data features.
    - y_dev: Development data labels.

    Returns:
    - A dictionary containing the parameters, the trained model, and the evaluation score.
    """
    print(f"Training with parameters: {params}")
    current_model = clone(model).set_params(**params)
    start_time_training = time()
    current_model.fit(X_train, y_train)
    end_time_training = time()
    training_time = end_time_training - start_time_training
    predictions = current_model.predict(X_dev)
    score = accuracy_score(y_dev, predictions)
    print(f"Finished training with parameters: {params}")
    return {
        "params": params,
        "model": current_model,
        "score": score,
        "time": training_time
    }

class ModelTrainerBase(GeneralModelTrainerBase):
    def __init__(self, model_name):
        super().__init__(model_name)

    def train_model(self, X_train, X_dev, y_train, y_dev, vectorizer, pca, label_encoder, model_folder,
                    multiclass=False, parameter_list=None):
        pass

    def perform_param_search(self, model_folder, model, param_grid, X_train, y_train, X_dev, y_dev, multiclass=False):
        """
        Perform a parameter search by training and evaluating models with each combination of parameters.

        Args:
        - model_folder: Folder to save the models.
        - model: The base model to be trained.
        - param_grid: The grid of parameters to search.
        - X_train: Training data features.
        - y_train: Training data labels.
        - X_dev: Development data features.
        - y_dev: Development data labels.
        - multiclass: Flag indicating if the problem is multiclass.

        Returns:
        - Best parameters, best score, best model, worst parameters, worst score, worst model, all models, training time, and parameter search time.
        """
        print(len(X_train), len(X_dev))
        print(len(y_train), len(y_dev))
        n_cpus = range(3, 36) # range(40, 73)  # CPUs to use

        pid = os.getpid()
        print("PID: %i" % pid)

        # Control which CPUs are made available for this script
        cpu_arg = ','.join([str(ci) for ci in n_cpus])
        cmd = 'taskset -cp %s %i' % (cpu_arg, pid)
        print("Executing command '%s' ..." % cmd)
        os.system(cmd)

        param_combinations = list(ParameterGrid(param_grid))
        all_params = []
        all_scores = []
        best_score = -np.inf
        worst_score = np.inf
        best_model = None
        worst_model = None
        best_params = None
        worst_params = None
        training_time = None
        models = {}

        start_time_param_search = time()

        # Parallel execution of model training and evaluation
        with ThreadPoolExecutor(max_workers=len(n_cpus)) as executor:
            futures = {
                executor.submit(train_and_evaluate, model, params, X_train, y_train, X_dev, y_dev): idx
                for idx, params in enumerate(param_combinations)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    params = result["params"]
                    score = result["score"]
                    current_model = result["model"]
                    train_time = result["time"]

                    all_params.append(params)
                    all_scores.append(score)

                    models[idx] = {
                        "params": params,
                        "model": current_model,
                        "score": score
                    }

                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_model = current_model
                        training_time = train_time

                    if score < worst_score:
                        worst_score = score
                        worst_params = params
                        worst_model = current_model

                except Exception as e:
                    print(f"Model training failed for index {idx} with params {params}: {e}")

        end_time_param_search = time()
        training_time_param_search = end_time_param_search - start_time_param_search
        print(f"Finished parameter search in {format_time(training_time_param_search)}.")

        grid_search_results = {
            "all_params": all_params,
            "all_scores": all_scores,
            "indexes": list(range(len(param_combinations)))
        }
        grid_search_path = os.path.join(model_folder, f"{self.model_name}_grid_search_results.json")
        with open(grid_search_path, "w") as f:
            json.dump(grid_search_results, f, indent=4)

        return best_params, best_score, best_model, worst_params, worst_score, worst_model, models, training_time, training_time_param_search

    def train_model_wrapper(self, df1_name, df2_name, data_type, reduction=None, flip=None):
        for v_name, model_folder in VECTORIZER.items():
            if data_type == "benchmark":
                if reduction:
                    model_folder = os.path.join(model_folder, f"{df1_name}_{int(reduction * 100)}")
                elif flip:
                    model_folder = os.path.join(model_folder, f"{df1_name}_{int(flip * 100)}")
                else:
                    model_folder = os.path.join(model_folder, f"{df1_name}")
                X_train, X_dev, y_train, y_dev, vectorizer, pca, label_encoder = prepare_data(
                    sentence_transformer=v_name,
                    df1_name=df1_name,
                    path=model_folder,
                    data_type=data_type,
                    reduction=reduction,
                    flip=flip)
            else:  # multi
                model_folder = os.path.join(model_folder, "multiclass")
                X_train, X_dev, y_train, y_dev, vectorizer, pca, label_encoder = prepare_data(
                    sentence_transformer=v_name,
                    path=model_folder,
                    data_type=data_type,
                    reduction=reduction,
                    flip=flip)

            multiclass = True if data_type == 'multi' or df1_name == 'Chemprot_c' else False
            if MODEL_TYPE != "baseline" and MODEL_TYPE != "multi":
                parameter_list=load_best_parameters("traditional", v_name, df1_name)
            else:
                parameter_list = None
            self.train_model(X_train, X_dev, y_train, y_dev, vectorizer, pca, label_encoder, model_folder,
                             multiclass=multiclass, parameter_list=parameter_list)

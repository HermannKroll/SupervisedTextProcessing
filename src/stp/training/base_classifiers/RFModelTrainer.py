import os
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_THREAD_LIMIT'] = '32'
os.environ['JOBLIB_MULTIPROCESSING'] = '1'

import joblib
from sklearn.ensemble import RandomForestClassifier
from time import time

from stp.run_config import RETRAIN_MODELS, PARAM_GRIDS, TRAIN_DEFAULT_MODELS
from stp.training.base_classifiers.ModelTrainerBase import ModelTrainerBase
from stp.training.GeneralModelTrainerBase import format_time


class RFModelTrainer(ModelTrainerBase):
    def __init__(self):
        super().__init__("rf")

    def train_model(self, X_train, X_dev, y_train, y_dev, vectorizer, pca, label_encoder, model_folder,
                    multiclass=False, parameter_list=None):
        worst_model_path = os.path.join(model_folder, "rf_model_worst.joblib")
        model_path = os.path.join(model_folder, "rf_model.joblib")
        default_model_path = os.path.join(model_folder, "rf_default_model.joblib")
        runtime_path = os.path.join(model_folder, "rf_runtime.txt")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if not os.path.exists(model_path) or RETRAIN_MODELS:
            rf = RandomForestClassifier()
            print("Performing parameter search for RandomForest model.")
            best_params, best_score, best_model, worst_params, worst_score, worst_model, models, training_time, training_time_param_search = self.perform_param_search(
                model_folder, rf, PARAM_GRIDS["RandomForestClassifier"], X_train, y_train, X_dev, y_dev, multiclass)

            joblib.dump((vectorizer, pca, label_encoder, worst_model), worst_model_path)
            print(f"Saved worst RandomForest model to {worst_model_path}")
            joblib.dump((vectorizer, pca, label_encoder, best_model), model_path)
            print(f"Saved best RandomForest model to {model_path}")

            self.save_performance(model_folder, best_params, best_score, worst_params, worst_score)

            total_time = training_time + training_time_param_search
            with open(runtime_path, 'w') as f:
                f.write(f"Training time: {format_time(training_time)}\n")
                f.write(f"Time for parameter search: {format_time(training_time_param_search)}\n")
                f.write(f"Total training time: {format_time(total_time)}\n")
        if TRAIN_DEFAULT_MODELS:
            print("Training Random Forest model with default parameters.")
            rf = RandomForestClassifier()
            start_time_training_default = time()
            rf.fit(X_train, y_train)
            end_time_training_default = time()
            training_time_default = end_time_training_default - start_time_training_default

            joblib.dump((vectorizer, pca, label_encoder, rf), default_model_path)
            print(f"Saved default Random Forest model to {default_model_path}")

            with open(runtime_path, 'a') as f:
                f.write(f"Training time for default model: {format_time(training_time_default)}\n")
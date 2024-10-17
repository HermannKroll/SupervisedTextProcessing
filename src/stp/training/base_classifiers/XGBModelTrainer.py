import os
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_THREAD_LIMIT'] = '32'
os.environ['JOBLIB_MULTIPROCESSING'] = '1'

import joblib
from xgboost import XGBClassifier
from time import time

from stp.run_config import RETRAIN_MODELS, PARAM_GRIDS, TRAIN_DEFAULT_MODELS, SAVE_ALL_MODELS
from stp.training.base_classifiers.ModelTrainerBase import ModelTrainerBase
from stp.training.GeneralModelTrainerBase import format_time


class XGBModelTrainer(ModelTrainerBase):
    def __init__(self):
        super().__init__("xgb")

    def train_model(self, X_train, X_dev, y_train, y_dev, pca, vectorizer, label_encoder, model_folder,
                    multiclass=False, parameter_list=None):
        worst_model_path = os.path.join(model_folder, "xgb_model_worst.joblib")
        model_path = os.path.join(model_folder, "xgb_model.joblib")
        default_model_path = os.path.join(model_folder, "xgb_default_model.joblib")
        runtime_path = os.path.join(model_folder, "xgb_runtime.txt")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if not os.path.exists(model_path) or RETRAIN_MODELS:
            xgb = XGBClassifier()

            if not parameter_list:
                print("Performing parameter search for XGB model.")
                best_params, best_score, best_model, worst_params, worst_score, worst_model, models, training_time, training_time_param_search = self.perform_param_search(
                    model_folder, xgb, PARAM_GRIDS["XGBClassifier"], X_train, y_train, X_dev, y_dev, multiclass)

                joblib.dump((vectorizer, pca, label_encoder, worst_model), worst_model_path)
                print(f"Saved worst XGBoost model to {worst_model_path}")
                joblib.dump((vectorizer, pca, label_encoder, best_model), model_path)
                print(f"Saved best XGBoost model to {model_path}")
                if SAVE_ALL_MODELS:
                    for idx, model_info in models.items():
                        model_save_path = os.path.join(model_folder, f"xgb_model_{idx}.joblib")
                        joblib.dump((vectorizer, pca, label_encoder, model_info["model"]), model_save_path)
                        print(f"Model saved at {model_save_path}")

                self.save_performance(model_folder, best_params, best_score, worst_params, worst_score)

                total_time = training_time + training_time_param_search
                with open(runtime_path, 'w') as f:
                    f.write(f"Training time: {format_time(training_time)}\n")
                    f.write(f"Time for parameter search: {format_time(training_time_param_search)}\n")
                    f.write(f"Total training time: {format_time(total_time)}\n")
            else:
                print("Training XGBoost model with predefined parameters without parameter search.")
                xgb.set_params(**parameter_list)

                start_time_training = time()
                xgb.fit(X_train, y_train)
                end_time_training = time()
                training_time = end_time_training - start_time_training

                joblib.dump((vectorizer, pca, label_encoder, xgb), model_path)
                print(f"Saved XGBoost model to {model_path}")

                with open(runtime_path, 'w') as f:
                    f.write(f"Training time: {format_time(training_time)}\n")
        if TRAIN_DEFAULT_MODELS:
            print("Training XGBoost model with default parameters.")
            xgb = XGBClassifier()
            start_time_training_default = time()
            xgb.fit(X_train, y_train)
            end_time_training_default = time()
            training_time_default = end_time_training_default - start_time_training_default

            joblib.dump((vectorizer, pca, label_encoder, xgb), default_model_path)
            print(f"Saved default XGBoost model to {default_model_path}")

            with open(runtime_path, 'a') as f:
                f.write(f"Training time for default model: {format_time(training_time_default)}\n")
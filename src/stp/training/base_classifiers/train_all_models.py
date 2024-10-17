import os
os.environ['OMP_THREAD_LIMIT'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['VECLIB_MAXIMUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['JOBLIB_MULTIPROCESSING'] = '1'
import shutil

from stp.training.base_classifiers.RFModelTrainer import RFModelTrainer
from stp.training.base_classifiers.SVCModelTrainer import SVCModelTrainer
from stp.training.base_classifiers.XGBModelTrainer import XGBModelTrainer
from stp.config import TRAINING_DATA_DIR

if __name__ == "__main__":
    # if os.path.exists(TRAINING_DATA_DIR):
    #     shutil.rmtree(TRAINING_DATA_DIR)
    #     os.makedirs(TRAINING_DATA_DIR)

    trainers = [RFModelTrainer(), XGBModelTrainer(), SVCModelTrainer()]

    for trainer in trainers:
        trainer.train_all_methods()
        # trainer.train_benchmark_model()

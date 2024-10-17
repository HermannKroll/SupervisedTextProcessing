from stp.training.bert.GenericBertModelTrainer import GenericBertModelTrainer
from stp.run_config import BERT_MODEL_MAPPING

if __name__ == "__main__":

    for model in BERT_MODEL_MAPPING.keys():
        trainer = GenericBertModelTrainer(model)
        # trainer.train_benchmark_model()
        trainer.train_chemprot_c_model()
        # trainer.train_all_labeling_methods()

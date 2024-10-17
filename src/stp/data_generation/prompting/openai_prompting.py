import json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from openai import OpenAI

from stp.config import OPENAI_INPUT_PATH, OPENAI_OUTPUT_PATH, OPENAI_METADATA_PATH
from stp.run_config import OPENAI_TOKEN


class OpenaiPromptingModel:
    _instance = None
    client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        self.client = OpenAI(api_key=OPENAI_TOKEN)

    @staticmethod
    def save_metadata(datasource, batch_input_file_id=None, batch_job_id=None, batch_output_file_id=None):
        metadata_path = os.path.join(OPENAI_METADATA_PATH, f"{datasource}_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        if batch_input_file_id is not None:
            metadata["batch_input_file_id"] = batch_input_file_id
        if batch_job_id is not None:
            metadata["batch_job_id"] = batch_job_id
        if batch_output_file_id is not None:
            metadata["batch_output_file_id"] = batch_output_file_id

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def load_metadata(self, datasource):
        metadata_path = os.path.join(OPENAI_METADATA_PATH, f"{datasource}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        return {}

    def answer_question(self, datasource):

        batch_input_file = self.client.files.create(
            file=open(os.path.join(OPENAI_INPUT_PATH, f"{datasource}.batchinput.jsonl"), "rb"),
            purpose="batch"
        )

        print(batch_input_file)

        batch_input_file_id = batch_input_file.id

        batch_job = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        batch_job_id = batch_job.id
        self.save_metadata(datasource, batch_input_file_id=batch_input_file_id, batch_job_id=batch_job_id)

    def check_status(self, datasource):
        # print(self.client.batches.list(limit=10))
        metadata = self.load_metadata(datasource=datasource)
        if "batch_job_id" in metadata:
            batch_job = self.client.batches.retrieve(metadata["batch_job_id"])
            print(batch_job.status)
            batch_output_file_id = batch_job.output_file_id
            self.save_metadata(datasource, batch_output_file_id=batch_output_file_id)
        else:
            print("No batch_input_file_id found in metadata")
        # print(self.client.files.list())

    def save_results(self, datasource):
        metadata = self.load_metadata(datasource=datasource)
        if "batch_output_file_id" in metadata:
            file_response = self.client.files.content(metadata["batch_output_file_id"])
            with open(os.path.join(OPENAI_OUTPUT_PATH, f"{datasource}.batchoutput.jsonl"), "w") as file:
                file.write(file_response.text)
        else:
            print("No batch_output_file_id found in metadata")


def analyze_openai_results():
    from stp.config import OPENAI_SENTENCES_PATH
    import pandas as pd
    from collections import defaultdict
    for benchmark in ["CDR", "Chemprot_c", "Chemprot", "DDI"]:
        df = pd.read_csv(os.path.join(OPENAI_SENTENCES_PATH, f"{benchmark}.relabeled.csv"))

        answer_dict = defaultdict(int)
        for answer in df["answer_word"]:
            answer_dict[answer] += 1

        print(benchmark)
        for answer, answer_count in answer_dict.items():
            print(answer, answer_count)
        print()


if __name__ == "__main__":
    benchmark_str = "Chemprot"
    model = OpenaiPromptingModel()
    # model.answer_question(benchmark_str)
    model.check_status(benchmark_str)
    # model.save_results(benchmark_str)

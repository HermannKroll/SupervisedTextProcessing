import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from stp.config import LM_CACHE_DIR
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.prompting.util import PROMPT_PATTERNS_BY_BENCHMARK
from stp.run_config import HUGGINGFACE_TOKEN

login(token=HUGGINGFACE_TOKEN)


class LlamaPromptingModel:
    _instance = None
    tokenizer = None
    model = None
    device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=LM_CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=LM_CACHE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}", torch.cuda.get_device_name(), torch.cuda.current_device())
        self.model.to(self.device)

    def answer_question(self, question):
        start_time = time.time()
        chat = [
            {"role": "user", "content": f"{question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(self.device)
        output = self.model.generate(inputs, max_new_tokens=1, do_sample=True, top_k=50, top_p=0.95,
                                     pad_token_id=self.tokenizer.eos_token_id, )
        answer = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        end_time = time.time()
        return answer[0], end_time - start_time


if __name__ == "__main__":
    lm = LlamaPromptingModel()

    gen = BaselineDataGeneration("Chemprot")
    prompt_patters = PROMPT_PATTERNS_BY_BENCHMARK["Chemprot"]

    for sentence, entities, label, sid in tqdm.tqdm(gen.sentences[:1].itertuples(index=False, name=None),
                                                    total=len(gen.sentences)):
        for idx, pattern in enumerate(prompt_patters[:1]):
            query = pattern.format(sentence, entities[0][0], entities[1][0])
            answer, response_time = lm.answer_question(query)
            print(answer.split("assistant")[-1].strip("<>|,.").strip().lower())

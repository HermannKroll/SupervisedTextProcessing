import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from stp.config import LM_CACHE_DIR
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.prompting.util import PROMPT_PATTERNS_BY_BENCHMARK
from stp.run_config import HUGGINGFACE_TOKEN

login(token=HUGGINGFACE_TOKEN)


class BioMistralPromptingModel:
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
        self.tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B", cache_dir=LM_CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B", cache_dir=LM_CACHE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")
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
    bm = BioMistralPromptingModel()

    gen = BaselineDataGeneration("Chemprot")
    prompt_patters = PROMPT_PATTERNS_BY_BENCHMARK["Chemprot"]

    for sentence, entities, label, sid in gen.sentences[:5].itertuples(index=False, name=None):
        for idx, pattern in enumerate(prompt_patters):
            query = pattern.format(sentence, entities[0][0], entities[1][0])
            answer, response_time = bm.answer_question(query)
            print(answer.split("[/INST]")[-1].strip("<>|,.").strip().lower())

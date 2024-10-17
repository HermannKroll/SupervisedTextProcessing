import json
import os

from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.chemprot_generation import ChemprotDataGeneration
from stp.data_generation.prompting.util import PROMPT_PATTERNS_BY_BENCHMARK
from stp.config import OPENAI_INPUT_PATH
from stp.benchmark import BENCHMARK_TO_NAME

GPT_MODEL = "gpt-4o"

BATCH_REQUEST_LINE = {
    "custom_id": None,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body":
        {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": None}],
            "max_tokens": 1
        }
}


def prepare_batch_data_file(data_source):
    if data_source == "Chemprot_c":
        gen = ChemprotDataGeneration(use_obfuscation=False, ignore_label_duplication=True)
    else:
        gen = BaselineDataGeneration(data_source, use_obfuscation=False)

    prompt_patters = PROMPT_PATTERNS_BY_BENCHMARK[data_source]

    rows = list()

    for sentence, entities, label, id in gen.sentences.itertuples(name=None, index=False):
        for idx, pattern in enumerate(prompt_patters):
            query = pattern.format(sentence.rstrip("."), entities[0][0], entities[1][0])
            json_data = BATCH_REQUEST_LINE
            json_data["body"]["messages"][0]["content"] = query
            json_data["custom_id"] = id + "-" + str(idx) if data_source != "Chemprot_c" else id + "-" + str(idx) + '-complex'
            rows.append(json.dumps(json_data))
    print(f"{data_source} prepared {len(rows)} batches")
    if data_source == "DDI":
        with open(os.path.join(OPENAI_INPUT_PATH, f"{data_source}.batchinput1.jsonl"), "wt") as f:
            f.write("\n".join(rows[:50000]))
        with open(os.path.join(OPENAI_INPUT_PATH, f"{data_source}.batchinput2.jsonl"), "wt") as f:
            f.write("\n".join(rows[50000:]))
    else:
        assert len(rows) <= 50000
        with open(os.path.join(OPENAI_INPUT_PATH, f"{data_source}.batchinput.jsonl"), "wt") as f:
            f.write("\n".join(rows))

if __name__ == "__main__":
    # for datasource in BENCHMARK_TO_NAME:
    #     prepare_batch_data_file(datasource)
    prepare_batch_data_file("Chemprot_c")

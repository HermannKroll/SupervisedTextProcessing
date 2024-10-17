import ast
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
from tqdm import tqdm

from stp.benchmark import BenchmarkLabel, ChemprotLabel
from stp.config import OLMO_SENTENCES_PATH, LLAMA_SENTENCES_PATH, BIOMISTRAL_SENTENCES_PATH, OPENAI_OUTPUT_PATH, \
    OPENAI_SENTENCES_PATH
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.chemprot_generation import ChemprotDataGeneration
from stp.data_generation.prompting.biomistral_prompting import BioMistralPromptingModel
from stp.data_generation.prompting.llama_prompting import LlamaPromptingModel
from stp.data_generation.prompting.olmo_prompting import OLMoPromptingModel
from stp.data_generation.prompting.util import PROMPT_PATTERNS_BY_BENCHMARK, NOT_ANSWERED, evaluate_label, \
    prompt_majority_vote, answer_word_to_label
from stp.run_config import OBFUSCATE_ENTITY_TAGS
from stp.util.metrics import print_full_metrics, results_to_latex

BENCHMARKS = [
    "CDR",
    "Chemprot_c",
    "Chemprot",
    "DDI",
]

LANGUAGE_MODELS = {
    "OLMO": {
        "qa_model": OLMoPromptingModel,
        "split_token": "assistant",
        "out_path": OLMO_SENTENCES_PATH
    },
    "LLAMA": {
        "qa_model": LlamaPromptingModel,
        "split_token": "assistant",
        "out_path": LLAMA_SENTENCES_PATH
    },
    "BIOMISTRAL": {
        "qa_model": BioMistralPromptingModel,
        "split_token": "[/INST]",
        "out_path": BIOMISTRAL_SENTENCES_PATH
    },
    "OPENAI": {
        "out_path": OPENAI_SENTENCES_PATH,
    }
}

PROMPT_TYPE_TO_INDEX = {
    "first_only": 0,
    "one_yes": 1,
    "two_yes": 2,
    "three_yes": 3
}


def relabel_baseline(benchmark: str, qa_model, output_path: str, split_token: str):
    assert OBFUSCATE_ENTITY_TAGS is False
    columns = ["sentence", "entities", "label", "sid", "prompt_idx", "answer_word", "response_time"]
    if benchmark == "Chemprot_c":
        gen = ChemprotDataGeneration(verbose=False, ignore_label_duplication=True)
    else:
        gen = BaselineDataGeneration(benchmark, verbose=False)
    prompt_patters = PROMPT_PATTERNS_BY_BENCHMARK[benchmark]

    relabeled = list()
    for sentence, entities, label, sid in tqdm(gen.sentences.itertuples(index=False, name=None),
                                               total=len(gen.sentences), desc=f"Relabeling {output_path}/{benchmark}"):
        for idx, pattern in enumerate(prompt_patters):
            query = pattern.format(sentence.rstrip("."), entities[0][0], entities[1][0])

            answer, response_time = qa_model.answer_question(query)
            answer_word = answer.split(split_token)[-1].strip("<>|,.").strip().lower()

            label_new = answer_word_to_label(answer_word, benchmark)

            relabeled.append((sentence, entities, label_new, sid, idx, answer_word, response_time))
    relabeled_df = pd.DataFrame(relabeled, columns=columns)
    df_path = os.path.join(output_path, f"{benchmark}.relabeled.csv")
    relabeled_df.to_csv(df_path, index=False)


def relabel_baseline_openai(benchmark: str, output_path: str):
    columns = ["sentence", "entities", "label", "sid", "prompt_idx", "answer_word", "response_time"]
    tokens = 0
    if benchmark == "DDI":
        input_path1 = os.path.join(OPENAI_OUTPUT_PATH, f"{benchmark}.batchoutput1.jsonl")
        input_path2 = os.path.join(OPENAI_OUTPUT_PATH, f"{benchmark}.batchoutput2.jsonl")
        if not os.path.exists(input_path1) or not os.path.exists(input_path2):
            return

        with open(input_path1, "rt") as f:
            json_list = list(f)

        with open(input_path2, "rt") as f:
            json_list.extend(list(f))

    else:
        input_path = os.path.join(OPENAI_OUTPUT_PATH, f"{benchmark}.batchoutput.jsonl")
        if not os.path.exists(input_path):
            return

        with open(input_path, "rt") as f:
            json_list = list(f)

    sentence_to_result = dict()
    for json_row in json_list:
        data = json.loads(json_row)
        sid: str = data.get("custom_id", "").replace("-complex", "")

        sid, prompt_idx = sid.rsplit("-", 1)
        answer_word = data["response"]["body"]["choices"][0]["message"]["content"]
        tokens += data["response"]["body"]["usage"]["prompt_tokens"] + 1
        answer_word = answer_word.lower().strip()

        label_new = answer_word_to_label(answer_word, benchmark)

        if sid not in sentence_to_result:
            sentence_to_result[sid] = list()
        sentence_to_result[sid].append((label_new, answer_word, int(prompt_idx)))

    if benchmark == "Chemprot_c":
        gen = ChemprotDataGeneration(ignore_label_duplication=True, use_obfuscation=False)
    else:
        gen = BaselineDataGeneration(benchmark, use_obfuscation=False)

    # ["sentence", "entities", "label", "sid", "prompt_idx", "answer_word", "response_time"]
    rows = list()
    for sentence, entities, _, sid in gen.sentences.itertuples(index=False, name=None):
        prompt_results = sentence_to_result[sid]
        for label, answer_word, prompt_idx in prompt_results:
            rows.append((sentence, entities, label, sid, prompt_idx, answer_word, 0.0))

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(os.path.join(output_path, f"{benchmark}.relabeled.csv"), index=False)


def evaluate_relabeled_sentences(output_path: str, benchmark: str, verbose=False):
    df_path = os.path.join(output_path, f"{benchmark}.relabeled.csv")
    columns = ["sentence", "entities", "label", "id"]
    if not os.path.exists(df_path):
        print("skipped", benchmark)
        return dict()

    relabeled_df = pd.read_csv(df_path)

    if benchmark == "Chemprot_c":
        gen = ChemprotDataGeneration(ignore_label_duplication=True, verbose=verbose, use_obfuscation=False)
        benchmark_label_positive = {ChemprotLabel.UPREGULATOR, ChemprotLabel.DOWNREGULATOR}
    else:
        gen = BaselineDataGeneration(benchmark, verbose=verbose, use_obfuscation=False)
        benchmark_label_positive = BenchmarkLabel.to_value(benchmark)

    first_only = dict(tp=0, fp=0, fn=0, tn=0, na=0)
    first_only_time = 0.0
    first_only_rows = list()

    one_yes = dict(tp=0, fp=0, fn=0, tn=0, na=0)
    one_yes_time = 0.0
    one_yes_num_votes = 0
    one_yes_rows = list()

    two_yes = dict(tp=0, fp=0, fn=0, tn=0, na=0)
    two_yes_time = 0.0
    two_yes_num_votes = 0
    two_yes_rows = list()

    three_yes = dict(tp=0, fp=0, fn=0, tn=0, na=0)
    three_yes_time = 0.0
    three_yes_num_votes = 0
    three_yes_rows = list()

    labels = list()
    response_times = list()

    #  ["sentence", "entities", "label", "sid", "prompt_idx", "answer_word", "response_time"]
    for sentence, entities, label, sid, prompt_idx, _, response_time in relabeled_df.itertuples(index=False, name=None):
        labels.append(label)
        response_times.append(response_time)

        # skip the first two promps - evaluate for all 3 prompts only
        if prompt_idx < 2:
            continue

        old_label = gen.sentences[gen.sentences["id"] == sid]["label"].iloc[0]

        entity_list = ast.literal_eval(entities)
        obfuscated_sentence = sentence
        for idx, (entity, *_) in enumerate(entity_list, 1):
            obfuscated_sentence = obfuscated_sentence.replace(entity, f"<entity{idx}>")

        # first only
        evaluate_label(old_label, labels[0], first_only)
        if labels[0] != NOT_ANSWERED:
            first_only_time += float(response_times[0])
            first_only_rows.append((obfuscated_sentence, entities, int(labels[0]), sid))

        # three prompt one yes
        majority_vote, time_spent, num_votes = prompt_majority_vote(n=1, labels=labels, times=response_times,
                                                                    benchmark_label=benchmark_label_positive)
        evaluate_label(old_label, majority_vote, one_yes)
        one_yes_time += time_spent
        one_yes_num_votes += num_votes
        one_yes_rows.append((obfuscated_sentence, entities, int(majority_vote), sid))

        # three prompt two yes
        majority_vote, time_spent, num_votes = prompt_majority_vote(n=2, labels=labels, times=response_times,
                                                                    benchmark_label=benchmark_label_positive)
        evaluate_label(old_label, majority_vote, two_yes)
        two_yes_time += time_spent
        two_yes_num_votes += num_votes
        two_yes_rows.append((obfuscated_sentence, entities, int(majority_vote), sid))

        # three prompt three yes
        majority_vote, time_spent, num_votes = prompt_majority_vote(n=3, labels=labels, times=response_times,
                                                                    benchmark_label=benchmark_label_positive)
        evaluate_label(old_label, majority_vote, three_yes)
        three_yes_time += time_spent
        three_yes_num_votes += num_votes
        three_yes_rows.append((obfuscated_sentence, entities, int(majority_vote), sid))

        labels.clear()
        response_times.clear()

    if verbose:
        print_full_metrics(**first_only, time=first_only_time,
                           desc=f"FirstOnly: Relabeled OLMo {benchmark} train baseline")
        print_full_metrics(**one_yes, time=one_yes_time, num_votes=one_yes_num_votes,
                           desc=f"OneYes: Relabeled OLMo {benchmark} train baseline")
        print_full_metrics(**two_yes, time=two_yes_time, num_votes=two_yes_num_votes,
                           desc=f"TwoYes: Relabeled OLMo {benchmark} train baseline")
        print_full_metrics(**three_yes, time=three_yes_time, num_votes=three_yes_num_votes,
                           desc=f"ThreeYes: Relabeled OLMo {benchmark} train baseline")

    df_out_path = os.path.join(output_path, f"{benchmark}.{{}}.relabeled.csv")
    pd.DataFrame(first_only_rows, columns=columns).to_csv(df_out_path.format("first_only"), index=False)
    pd.DataFrame(one_yes_rows, columns=columns).to_csv(df_out_path.format("one_yes"), index=False)
    pd.DataFrame(two_yes_rows, columns=columns).to_csv(df_out_path.format("two_yes"), index=False)
    pd.DataFrame(three_yes_rows, columns=columns).to_csv(df_out_path.format("three_yes"), index=False)

    result = dict(
        first_only=dict(data=first_only, time=first_only_time),
        one_yes=dict(data=one_yes, time=one_yes_time, num_votes=one_yes_num_votes),
        two_yes=dict(data=two_yes, time=two_yes_time, num_votes=two_yes_num_votes),
        three_yes=dict(data=three_yes, time=three_yes_time, num_votes=three_yes_num_votes)
    )
    with open(os.path.join(output_path, f"{benchmark}.relabel.statistics.json"), "wt") as f:
        json.dump(result, f, indent=2)

    return result


def recreate_dataset(output_path: str, prompt_type: str, benchmark: str):
    assert prompt_type in PROMPT_TYPE_TO_INDEX.keys()
    index = PROMPT_TYPE_TO_INDEX[prompt_type]

    df = pd.read_csv(os.path.join(output_path, f"{benchmark}.relabeled.csv"))
    df = df[df["prompt_idx"] == index]

    df.to_csv(os.path.join(output_path, f"{benchmark}.{prompt_type}.relabeled.csv"))


if __name__ == "__main__":
    # import gc
    # for lm in LANGUAGE_MODELS:
    #     if lm == "OPENAI":
    #         continue
    #     qa_model = LANGUAGE_MODELS[lm]["qa_model"]()
    #     out_path = LANGUAGE_MODELS[lm]["out_path"]
    #     split_token = LANGUAGE_MODELS[lm]["split_token"]
    #     for bm in BENCHMARKS:
    #         relabel_baseline(bm, qa_model, out_path, split_token)
    #     qa_model.model = None
    #     qa_model.tokenizer = None
    #     qa_model.device = None
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # relabel_baseline_openai("CDR", OPENAI_SENTENCES_PATH)
    # relabel_baseline_openai("Chemprot_c", OPENAI_SENTENCES_PATH)
    # relabel_baseline_openai("Chemprot", OPENAI_SENTENCES_PATH)
    # relabel_baseline_openai("DDI", OPENAI_SENTENCES_PATH)

    for lm in LANGUAGE_MODELS:
        out_path = LANGUAGE_MODELS[lm]["out_path"]
        # analyze prompt answer words
        # print(lm)
        # for bm in BENCHMARKS:
        #     analyze_prompt_answers(out_path, bm)

        # evaluate relabeling
        results = list()
        for bm in BENCHMARKS:
            result = evaluate_relabeled_sentences(out_path, bm)
            results.append(result)
        results_to_latex(results, desc=lm)

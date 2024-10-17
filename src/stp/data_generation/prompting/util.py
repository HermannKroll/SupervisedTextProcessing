import os
from collections import defaultdict
from typing import Union

import pandas as pd

from stp.benchmark import BenchmarkLabel, ChemprotLabel

NOT_ANSWERED = -1

PROMPT_PATTERNS_BY_BENCHMARK = {
    "Chemprot": [
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} interacts "
            "with {2}? Interacts describes that {1} has a reaction with {2}, e.g., agonist-inhibitor, antagonist, "
            "indirect-downregulator, inhibitor, activator, agonist, agonist-activator or indirect-upregulator. Answer "
            "only with yes or no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} interacts "
            "with {2}? Interacts means that {1} activates, binds, regulates or blocks {2}. Answer only with yes or "
            "no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} interacts "
            "with {2}? Interacts means that {1} has an effect on {2}. Answer only with yes or no.")
    ],
    "Chemprot_c": [
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} regulates "
            "{2}? We ask for three kinds of regulations between {1} and {2}: (1) up regulation means that {1} is an "
            "activator, agonist, agonist-activator or indirect-upregulator of {2}. (2) down regulation means that {1} "
            "is an agonist-inhibitor, antagonist, indirect-downregulator or inhibitor of {2}. (3) no means that there "
            "is no regulation. Answer only with up, down or no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} regulates "
            "{2}? We ask for three kinds of regulations between {1} and {2}: (1) up regulation means that {1} "
            "activates, induces, upregulates or stimulates {2}. (2) down regulation means that {1} inhibits, "
            "downregulates or decreases {2}. (3) no means that there is no regulation. Answer only with up, down or "
            "no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} regulates "
            "{2}? We ask for three kinds of regulations between {1} and {2}: (1) up regulation means that {1} has a "
            "positive effect of {2}. (2) down regulation means that {1} has a negative effect of {2}. (3) no means "
            "that there is no regulation. Answer only with up, down or no.")
    ], "DDI": [
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} interacts "
            "with {2}. Interacts describes that {1} interacts with {2}, e.g., as a drug-drug interaction, via a "
            "shared target or via some mechanism. Answer only with yes or no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} interacts "
            "with {2}. Interacts means that {1} has a reaction with {2}. Answer only with yes or no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} interacts "
            "with {2}. Interacts means that {1} has an interaction with {2}. Answer only with yes or no.")
    ], "CDR": [
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} induces {2}? "
            "Induces describes that {1} causes or leads to {2}. Answer only with yes or no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} induces {2}? "
            "Induces describes that {2} is a side effect of {1}, an adverse effect of {1}, a toxic effect caused by "
            "{1}, or a complication caused by {1}. Answer only with yes or no."),
        (
            "Consider the following sentence: {0}. Does this sentence describes the information that {1} induces {2}? "
            "Induces means that {1} increases, stimulates, enhances, potentitates or activates the expression of {2}. "
            "Answer only with yes or no.")
    ]
}


def evaluate_label(old: int, new: int, votes: dict):
    # count not answered separately
    if new == NOT_ANSWERED:
        votes["na"] += 1

    # true_label positive and pred_label positive (any EQUAL positive class)
    if old != BenchmarkLabel.NEGATIVE and new == old:
        votes["tp"] += 1
    # wrong labels OR true_label positive but pred_label negative
    elif new == NOT_ANSWERED or (old != BenchmarkLabel.NEGATIVE and new == BenchmarkLabel.NEGATIVE):
        votes["fn"] += 1
    # true_label negative but pred_label positive
    elif old == BenchmarkLabel.NEGATIVE and new != BenchmarkLabel.NEGATIVE:
        votes["fp"] += 1
    # true_label negative and pred_label negative
    elif old == BenchmarkLabel.NEGATIVE and new == old:
        votes["tn"] += 1


def prompt_majority_vote(n: int, labels: list, times: list, benchmark_label: Union[int, set]):
    """
    Apply majority vote to the prompts with minimum n votes. After n votes, the voting
    gets stopped to reduce the required prompting time. In this case, the following
    prompt results are ignored.
    :param benchmark_label: the label of the benchmark being evaluated
    :param n: min number of votes
    :param labels: list of prompting labels
    :param times: list of prompting times
    :return: majority vote, cumulative voting time
    """
    if isinstance(benchmark_label, int):
        benchmark_label = {benchmark_label}

    yes_votes = {k: 0 for k in benchmark_label}
    num_votes = 0
    cumulative_response_time = 0.0

    if all(label == NOT_ANSWERED for label in labels):
        return NOT_ANSWERED, cumulative_response_time, num_votes

    vote = BenchmarkLabel.NEGATIVE
    for new_label, response_time in zip(labels, times):
        num_votes += 1
        cumulative_response_time += float(response_time)

        if new_label not in benchmark_label:
            continue

        # a valid vote for the current benchmark
        yes_votes[new_label] += 1

        # check if we got already enough votes
        majority_class = next(iter(k for k, v in yes_votes.items() if v >= n), None)
        if majority_class:
            vote = majority_class
            break
    return vote, cumulative_response_time, num_votes


def answer_word_to_label(answer_word, benchmark):
    if benchmark == "Chemprot_c":
        if answer_word == "no":
            label_new = ChemprotLabel.NEGATIVE
        elif answer_word == "up":
            label_new = ChemprotLabel.UPREGULATOR
        elif answer_word == "down":
            label_new = ChemprotLabel.DOWNREGULATOR
        else:
            label_new = NOT_ANSWERED
    else:
        if answer_word == "yes":
            label_new = BenchmarkLabel.to_value(benchmark)
        elif answer_word == "no":
            label_new = BenchmarkLabel.NEGATIVE
        else:
            label_new = NOT_ANSWERED
    return label_new


def analyze_prompt_answers(path, benchmark):
    path = os.path.join(path, f"{benchmark}.relabeled.csv")
    df = pd.read_csv(path)

    answer_dict = defaultdict(int)
    for answer in df["answer_word"]:
        answer_dict[answer] += 1

    print(benchmark, ", ".join(str(answer) + ": " + str(answer_count) for answer, answer_count in answer_dict.items()))

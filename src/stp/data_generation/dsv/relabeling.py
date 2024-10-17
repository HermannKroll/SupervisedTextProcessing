import os.path
import time

import pandas as pd

from stp.benchmark import BenchmarkLabel
from stp.config import DSV_SENTENCES_PATH
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.dsv.knowledgebase import DDIKnowledgeBase, CDRKnowledgeBase, ChemProtKnowledgeBase
from stp.util.metrics import print_full_metrics, calc_precision, calc_recall, calc_f1_score, calc_accuracy

BENCHMARK_TO_KNOWLEDGEBASE = {
    "DDI": DDIKnowledgeBase,
    "CDR": CDRKnowledgeBase,
    "Chemprot": ChemProtKnowledgeBase
}


def relabel_baseline_with_dsv():
    relabel_time = 0.0
    num_sentences = 0
    results = list()
    for benchmark in ["CDR", "Chemprot_c", "Chemprot", "DDI"]:
        if benchmark == "Chemprot_c":
            results.extend(["-"] * 5)
            continue

        found_mesh_relations = 0
        found_synonym_relations = 0
        if benchmark == "CDR":
            gen = BaselineDataGeneration(data_source=benchmark, verbose=False, store_entity_id=True)
        else:
            gen = BaselineDataGeneration(data_source=benchmark, verbose=False)
        kb = BENCHMARK_TO_KNOWLEDGEBASE[benchmark]()
        relabeled = list()

        start_time = time.time()

        for sentence, entities, label, id in gen.sentences.itertuples(name=None, index=False):
            subject, object = entities[0][0].lower(), entities[1][0].lower()

            if (subject in kb.interactions and object in kb.interactions[subject]
                    or object in kb.interactions and subject in kb.interactions[object]):
                new_label = BenchmarkLabel.to_value(gen.data_source)

            # for CDR check also for the mesh ids
            elif benchmark == "CDR":
                subject_id = entities[0][2].lower()
                object_id = entities[1][2].lower()
                if kb.check_by_id(subject, object, subject_id, object_id):
                    found_mesh_relations += 1
                    new_label = BenchmarkLabel.to_value(gen.data_source)
                else:
                    new_label = BenchmarkLabel.NEGATIVE

            # for DDI check also for synonyms
            elif benchmark == "DDI":
                if kb.check_by_synonyms(subject, object):
                    new_label = BenchmarkLabel.to_value(gen.data_source)
                    found_synonym_relations += 1
                else:
                    new_label = BenchmarkLabel.NEGATIVE

            # no interaction found
            else:
                new_label = BenchmarkLabel.NEGATIVE

            relabeled.append((sentence, entities, new_label, id))

        relabel_time += time.time() - start_time
        num_sentences += len(gen.sentences)

        print(benchmark, f"{found_mesh_relations=}")
        print(benchmark, f"{found_synonym_relations=}")

        relabeled_df = pd.DataFrame(relabeled, index=None, columns=["sentence", "entities", "label", "id"])
        with open(os.path.join(DSV_SENTENCES_PATH, f"{benchmark}.relabeled.csv"), "wt") as f:
            relabeled_df.to_csv(f, index=False)

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for old_label, new_label in zip(gen.sentences["label"], relabeled_df["label"]):

            if old_label == BenchmarkLabel.to_value(gen.data_source) and new_label == old_label:
                tp += 1
            elif old_label == BenchmarkLabel.to_value(gen.data_source) and new_label == BenchmarkLabel.NEGATIVE:
                fn += 1
            elif old_label == BenchmarkLabel.NEGATIVE and new_label == BenchmarkLabel.to_value(gen.data_source):
                fp += 1
            elif old_label == BenchmarkLabel.NEGATIVE and new_label == old_label:
                tn += 1
            else:
                raise Exception("unexpected outcome")

        print_full_metrics(tp, fp, fn, tn, f"Relabeled dsv {benchmark} train baseline")
        precision = calc_precision(tp, fp)
        recall = calc_recall(tp, fn)
        f1 = calc_f1_score(precision, recall)
        accuracy = calc_accuracy(tp, tn, fn, fp)
        results.extend([round(precision, 2), round(recall, 2), round(f1, 2), round(accuracy, 2), "-"])
    print("Relabeling took {} s for {} sentences\n"
          "--> {} s/sentence"
          .format(relabel_time,
                  num_sentences,
                  relabel_time / num_sentences))

    results.insert(0, round(relabel_time / num_sentences, 2))

    # print latex code
    print(" & ".join(str(r) for r in results))


if __name__ == "__main__":
    relabel_baseline_with_dsv()

import itertools

import pandas as pd

from stp.benchmark import ChemprotLabel
from stp.data_generation.baseline_generation import BaselineDataGeneration
from stp.data_generation.util import clean_entity_type
from stp.run_config import BENCHMARKS, OBFUSCATE_ENTITY_TAGS

BENCHMARK_RUN = "train"
DATA_SOURCE = "Chemprot"

CHEMPROT_CLASS_MAPPING = {
    0: ChemprotLabel.NEGATIVE,       # Negative
    3: ChemprotLabel.DOWNREGULATOR,  # Downregulator
    4: ChemprotLabel.UPREGULATOR,    # Upregulator
    5: ChemprotLabel.DOWNREGULATOR,  # Agonist
    6: ChemprotLabel.UPREGULATOR,    # Antagonist
    9: ChemprotLabel.NEGATIVE        # Substrate
}


class ChemprotDataGeneration(BaselineDataGeneration):
    def __init__(self, run=BENCHMARK_RUN, use_obfuscation=OBFUSCATE_ENTITY_TAGS, verbose=True,
                 ignore_label_duplication=False):
        self.run = run
        self.ignore_label_duplication = ignore_label_duplication
        super().__init__(data_source=DATA_SOURCE, use_obfuscation=use_obfuscation, run=run, verbose=verbose)

    def __repr__(self):
        return (f"ChemprotGeneration: {self.data_source}.{self.run}:"
                f"\n\t (0) {len(self.sentences[self.sentences['label'] == ChemprotLabel.NEGATIVE])}"
                f"\n\t (3) {len(self.sentences[self.sentences['label'] == ChemprotLabel.UPREGULATOR])}"
                f"\n\t (4) {len(self.sentences[self.sentences['label'] == ChemprotLabel.DOWNREGULATOR])}")

    def _generate_data(self):
        labeled_sentences = list()
        default_class_sentences = list()
        benchmark = BENCHMARKS[self.data_source]

        for document in benchmark.documents[self.run]:
            for sentence in document.sentences:
                duplicate_index = 0
                for t1, t2 in itertools.combinations(sentence.relation_tags, 2):
                    t1_ent_type = clean_entity_type(t1.ent_type)
                    t2_ent_type = clean_entity_type(t2.ent_type)

                    # skip invalid combinations of entity type pairs
                    if ((t1_ent_type, t2_ent_type) not in document.relation_types
                            and (t2_ent_type, t1_ent_type) not in document.relation_types):
                        continue

                    if (t1.ent_id, t2.ent_id) in document.relation2subclass:
                        labels = {CHEMPROT_CLASS_MAPPING[label] for label in
                                   document.relation2subclass[(t1.ent_id, t2.ent_id)]}
                        old_labels = document.relation2subclass[(t1.ent_id, t2.ent_id)]
                    elif (t2.ent_id, t1.ent_id) in document.relation2subclass:
                        labels = {CHEMPROT_CLASS_MAPPING[label] for label in
                                   document.relation2subclass[(t2.ent_id, t1.ent_id)]}
                        old_labels = document.relation2subclass[(t2.ent_id, t1.ent_id)]
                    else:
                        labels = {ChemprotLabel.NEGATIVE}
                        old_labels = {ChemprotLabel.NEGATIVE}

                    entities = [(t1.text, clean_entity_type(t1.ent_type)), (t2.text, clean_entity_type(t2.ent_type))]
                    for label in labels:
                        labeled_sentences.append((sentence.text, entities, label,
                                                  sentence.id + "-" + str(duplicate_index)))
                        if self.ignore_label_duplication:
                            break

                    for label in old_labels:
                        default_class_sentences.append((sentence.text, entities, label,
                                                              sentence.id + "-" + str(duplicate_index)))
                    duplicate_index += 1

        columns = ['sentence', 'entities', 'label', 'id']
        self.sentences = pd.DataFrame(labeled_sentences, columns=columns)
        self.default_class_sentences = pd.DataFrame(default_class_sentences, columns=columns)
        self._apply_transformation_policies()


def default_class_distribution():
    cgen = ChemprotDataGeneration()
    # print the class distribution of the true (not deflated) annotated classes
    import collections
    label_count = collections.defaultdict(int)
    for label in cgen.default_class_sentences["label"]:
        label_count[label] += 1

    for label in label_count:
        print(label, label_count[label])


if __name__ == "__main__":
    gen = ChemprotDataGeneration()
    print(gen)
    default_class_distribution()


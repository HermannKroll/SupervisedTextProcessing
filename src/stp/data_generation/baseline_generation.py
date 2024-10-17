import itertools
from copy import deepcopy

import pandas as pd

from stp.benchmark import BenchmarkLabel
from stp.data_generation.base import DataGeneration
from stp.data_generation.util import clean_entity_type
from stp.document import Document
from stp.run_config import OBFUSCATE_ENTITY_TAGS, BENCHMARKS

BENCHMARK_RUN = "train"


class BaselineDataGeneration(DataGeneration):
    def __init__(self, data_source, run=BENCHMARK_RUN, use_obfuscation=OBFUSCATE_ENTITY_TAGS, verbose=True,
                 store_entity_id=False):
        self.run = run
        self.store_entity_id = store_entity_id
        super().__init__(data_source, use_obfuscation=use_obfuscation, verbose=verbose)

    def __repr__(self):
        return f"BaselineGenerator: {self.data_source}.{self.run}: [{self.negatives()} (-)|{self.positives()} (+)]"

    def __str__(self):
        return self.__repr__()

    def _raw_sentences_analysis(self, document: Document, labeled_sentences: list):
        duplicate_index = 0
        for sentence in document.sentences:
            for t1, t2 in itertools.combinations(sentence.relation_tags, 2):
                t1_ent_type = clean_entity_type(t1.ent_type)
                t2_ent_type = clean_entity_type(t2.ent_type)

                # skip invalid combinations of entity type pairs
                if ((t1_ent_type, t2_ent_type) not in document.relation_types
                        and (t2_ent_type, t1_ent_type) not in document.relation_types):
                    continue

                # the raw approach ignores false-labeled combinations
                if (t1.ent_id, t2.ent_id) in document.relations:
                    label = BenchmarkLabel.to_value(self.data_source)
                    subject = t1
                    object = t2
                elif (t2.ent_id, t1.ent_id) in document.relations:
                    label = BenchmarkLabel.to_value(self.data_source)
                    subject = t2
                    object = t1
                else:
                    subject = t1
                    object = t2
                    label = BenchmarkLabel.NEGATIVE

                labeled_sentences.append((sentence.text, [(subject.text, subject.ent_type),
                                                          (object.text, object.ent_type)],
                                          label, sentence.id + "-" + str(duplicate_index)))
                duplicate_index += 1

    def _obfuscated_sentences_analysis(self, document: Document, labeled_sentences: list):
        for sentence in document.sentences:
            duplicate_index = 0
            for t1, t2 in itertools.combinations(sentence.relation_tags, 2):
                t1_ent_type = clean_entity_type(t1.ent_type)
                t2_ent_type = clean_entity_type(t2.ent_type)

                # skip invalid combinations of entity type pairs
                if ((t1_ent_type, t2_ent_type) not in document.relation_types
                        and (t2_ent_type, t1_ent_type) not in document.relation_types):
                    continue

                label = BenchmarkLabel.NEGATIVE
                if (t1.ent_id, t2.ent_id) in document.relations or (t2.ent_id, t1.ent_id) in document.relations:
                    label = BenchmarkLabel.to_value(self.data_source)

                if self.store_entity_id:
                    entities = [(t1.text, clean_entity_type(t1.ent_type), t1.ent_id),
                                (t2.text, clean_entity_type(t2.ent_type), t2.ent_id)]
                else:
                    entities = [(t1.text, clean_entity_type(t1.ent_type)), (t2.text, clean_entity_type(t2.ent_type))]
                labeled_sentences.append((sentence.text, entities, label, sentence.id + "-" + str(duplicate_index)))
                duplicate_index += 1

    def _generate_data(self):
        labeled_sentences = list()
        benchmark = BENCHMARKS[self.data_source]

        self.relation_types = set()

        if self.run not in benchmark.documents:
            self.sentences = pd.DataFrame(columns=['sentence', 'entities', 'label', 'id'], index=None)
            return

        for document in benchmark.documents[self.run]:
            self.relation_types |= document.relation_types

            if self.use_obfuscation:
                self._obfuscated_sentences_analysis(document, labeled_sentences)
            else:
                self._raw_sentences_analysis(document, labeled_sentences)

        columns = ['sentence', 'entities', 'label', 'id']
        self.sentences = pd.DataFrame(labeled_sentences, columns=columns)
        self._apply_transformation_policies()

    def negatives(self):
        return len(self.sentences[self.sentences["label"] == BenchmarkLabel.NEGATIVE])

    def positives(self):
        return len(self.sentences[self.sentences["label"] != BenchmarkLabel.NEGATIVE])

    def split_data(self, random_state, split_ratio=0.5):
        self.sentences = self.sentences.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if split_ratio >= 0 and split_ratio <= 1:
            split_idx = int(len(self.sentences) * split_ratio)
            df_train = self.sentences.iloc[:split_idx]
            df_dev = self.sentences.iloc[split_idx:]
            gen_train = deepcopy(self)
            gen_dev = deepcopy(self)
            gen_train.sentences = df_train
            gen_dev.sentences = df_dev
            return gen_train, gen_dev
        else:
            raise ValueError("Split ratio must be between 0 and 1.")


if __name__ == '__main__':
    gen = BaselineDataGeneration("DDI")
    gen_train, gen_dev = gen.split_data(random_state=42)
    print("negative train", len(gen_train.sentences[gen_train.sentences["label"] == BenchmarkLabel.NEGATIVE]))
    print("positive train",
          len(gen_train.sentences[gen_train.sentences["label"] == BenchmarkLabel.to_value(gen.data_source)]))

    print("negative dev", len(gen_dev.sentences[gen_dev.sentences["label"] == BenchmarkLabel.NEGATIVE]))
    print("positive dev",
          len(gen_dev.sentences[gen_dev.sentences["label"] == BenchmarkLabel.to_value(gen.data_source)]))

    gen = BaselineDataGeneration("DDI")
    print("negative", gen.negatives())
    print("positive", gen.positives())
    print("relation_types", gen.relation_types)

    gen = BaselineDataGeneration("CDR")
    print("negative", gen.negatives())
    print("positive", gen.positives())
    print("relation_types", gen.relation_types)

    gen = BaselineDataGeneration("Chemprot")
    print("negative", gen.negatives())
    print("positive", gen.positives())
    print("relation_types", gen.relation_types)

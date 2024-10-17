import itertools
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from stp.config import CHEMPROT_DSV_KB_FILE, CDR_DSV_KB_FILE, DDI_DSV_KB_FILE
from stp.data_generation.dsv.vocabulary import ChemicalVocab, DiseaseVocab, GeneVocab, DrugbankVocabulary


class KnowledgeBase:
    def __init__(self, name):
        self.interactions = defaultdict(set)
        self.name = name
        self.prepare_data()

        print(f"{self.name}: {sum(len(i) for i in self.interactions.values())} known interactions")

    def prepare_data(self):
        raise NotImplementedError("Implement this method")


class CTDKnowledgeBase(KnowledgeBase):
    vocab_this = None
    vocab_other = None

    def __init__(self, path, name):
        self.path = path
        self.interactions_by_id = defaultdict(set)
        super().__init__(name=name)

    def prepare_data(self):
        with open(self.path) as f:
            df = pd.read_csv(f, comment='#', header=None, index_col=None)
        unknown_by_vocab = set()
        for row in tqdm(df.itertuples(index=False, name=None), total=len(df), desc=f"Preparing {self.name}"):
            this = str(row[0]).strip().lower()
            this_id = str(row[1]).split(":")[-1].strip().lower()
            other = str(row[3]).strip().lower()
            other_id = str(row[4]).split(":")[-1].strip().lower()

            if (this not in self.vocab_this.entities or
                    other not in self.vocab_other.entities):
                unknown_by_vocab.add(this + "|||" + other)

            self.interactions[this].add(other)
            self.interactions_by_id[this_id].add(other_id)

        with open(f"{self.name}.unknown.txt", "wt") as f:
            f.write("\n".join(unknown_by_vocab))


class CDRKnowledgeBase(CTDKnowledgeBase):
    def __init__(self):
        self.vocab_this = ChemicalVocab()
        self.vocab_other = DiseaseVocab()
        super().__init__(path=CDR_DSV_KB_FILE, name="CDR")

    def check_by_id(self, subject, object, subject_id, object_id):
        # test if the id-pair is known by the vocabulary (either way)
        if subject_id in self.interactions_by_id and object_id in self.interactions_by_id[subject_id]:
            return True
        if object_id in self.interactions_by_id and subject_id in self.interactions_by_id[object_id]:
            return True

        # expand names to synonyms to mesh ids to test if the entities are known by the vocabulary
        if (subject in self.vocab_this.synonym_to_mesh_ids
                and object in self.vocab_other.synonym_to_mesh_ids):
            subject_ids = self.vocab_this.synonym_to_mesh_ids[subject]
            object_ids = self.vocab_other.synonym_to_mesh_ids[object]
        elif (object in self.vocab_this.synonym_to_mesh_ids
              and subject in self.vocab_other.synonym_to_mesh_ids):
            subject_ids = self.vocab_this.synonym_to_mesh_ids[object]
            object_ids = self.vocab_other.synonym_to_mesh_ids[subject]
        else:
            return False

        known_interaction = False
        for subject_id, object_id in itertools.product(subject_ids, object_ids):
            # check if the translated mesh ids are in a known relation
            known_interaction = (subject_id in self.interactions_by_id
                                 and object_id in self.interactions_by_id[subject_id]
                                 or object_id in self.interactions_by_id
                                 and subject_id in self.interactions_by_id[object_id])
            if known_interaction:
                break
        return known_interaction


class ChemProtKnowledgeBase(CTDKnowledgeBase):
    def __init__(self):
        self.vocab_this = ChemicalVocab()
        self.vocab_other = GeneVocab()
        super().__init__(path=CHEMPROT_DSV_KB_FILE, name="ChemProt")


class DDIKnowledgeBase(KnowledgeBase):
    def __init__(self):
        super().__init__(name="DDI")
        self.vocab = DrugbankVocabulary()

    def prepare_data(self):

        root: ET.Element = ET.parse(DDI_DSV_KB_FILE).getroot()

        for drug in tqdm(root, desc=f"Preparing {self.name}"):
            name = drug.find("{http://www.drugbank.ca}name").text.strip().lower()
            interactions = drug.find("{http://www.drugbank.ca}drug-interactions")

            for interaction in interactions:
                other_name = interaction.find("{http://www.drugbank.ca}name").text.strip().lower()
                self.interactions[name].add(other_name)

    def check_by_synonyms(self, subject, object):
        # test if the entities are directly (or by synonym) known by the vocabulary
        if (subject not in self.vocab.entities
                or object not in self.vocab.entities):
            return False

        # resolve synonyms directly or via the base name
        if subject in self.vocab.name_to_synonyms:
            subject_synonyms = self.vocab.name_to_synonyms[subject]
        else:
            subject_base_name = self.vocab.synonym_to_name[subject]
            subject_synonyms = self.vocab.name_to_synonyms[subject_base_name]

        if object in self.vocab.name_to_synonyms:
            object_synonyms = self.vocab.name_to_synonyms[object]
        else:
            object_base_name = self.vocab.synonym_to_name[object]
            object_synonyms = self.vocab.name_to_synonyms[object_base_name]

        known_interaction = False
        for subject_id, object_id in itertools.product(subject_synonyms, object_synonyms):
            # check if the translated mesh ids are in a known relation
            known_interaction = (subject_id in self.interactions
                                 and object_id in self.interactions[subject_id]
                                 or object_id in self.interactions
                                 and subject_id in self.interactions[object_id])
            if known_interaction:
                break
        return known_interaction


if __name__ == "__main__":
    CDRKnowledgeBase()
    ChemProtKnowledgeBase()
    DDIKnowledgeBase()

from collections import defaultdict

import pandas as pd

from stp.config import CHEMICALS_VOCAB_FILE, DISEASES_VOCAB_FILE, GENES_VOCAB_FILE, DRUGBANK_VOCAB_FILE


def read_file(file):
    return pd.read_csv(file,
                       sep=',',
                       comment='#',
                       skipinitialspace=True,
                       skip_blank_lines=True,
                       quotechar='"',
                       keep_default_na=False,
                       low_memory=False)


class ChemicalVocab:
    _instance = None

    mesh_id_to_synonyms = dict()
    name_to_mesh_id = dict()
    mesh_id_to_name = dict()
    synonym_to_mesh_ids = defaultdict(set)
    entities = set()

    def __new__(cls, vocabulary_file=CHEMICALS_VOCAB_FILE):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = read_file(vocabulary_file)
            cls._instance.prepare()
        return cls._instance

    def prepare(self):
        # ChemicalName,ChemicalID,CasRN,Definition,ParentIDs,TreeNumbers,ParentTreeNumbers,Synonyms
        for name, mesh_id, _, _, _, _, _, synonyms in self.data.itertuples(name=None, index=False):
            name = name.lower()
            mesh_id = mesh_id.split(":")[-1].lower()

            synonyms = {s.lower() for s in str(synonyms).split("|") if s}
            synonyms.add(name)
            self.name_to_mesh_id[name] = mesh_id
            self.mesh_id_to_name[mesh_id] = name

            self.mesh_id_to_synonyms[mesh_id] = synonyms
            for s in synonyms:
                self.synonym_to_mesh_ids[s].add(mesh_id)

            self.entities |= synonyms


class DiseaseVocab:
    _instance = None

    mesh_id_to_synonyms = dict()
    name_to_mesh_id = dict()
    mesh_id_to_name = dict()
    synonym_to_mesh_ids = defaultdict(set)
    entities = set()

    def __new__(cls, vocabulary_file=DISEASES_VOCAB_FILE):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = read_file(vocabulary_file)
            cls._instance.prepare()
        return cls._instance

    def prepare(self):
        # DiseaseName,DiseaseID,AltDiseaseIDs,Definition,ParentIDs,TreeNumbers,ParentTreeNumbers,Synonyms,SlimMappings
        for name, mesh_id, _, _, _, _, _, synonyms, _ in self.data.itertuples(name=None, index=False):
            name = name.lower()
            mesh_id = mesh_id.split(":")[-1].lower()

            synonyms = {s.lower() for s in str(synonyms).split("|") if s}
            synonyms.add(name)
            self.mesh_id_to_name[mesh_id] = name
            self.name_to_mesh_id[name] = mesh_id

            self.mesh_id_to_synonyms[mesh_id] = synonyms
            for s in synonyms:
                self.synonym_to_mesh_ids[s].add(mesh_id)

            self.entities |= synonyms


class GeneVocab:
    _instance = None

    gene_id_to_synonyms = dict()
    gene_id_to_alternatives = dict()
    name_to_gene_id = dict()
    gene_id_to_name = dict()
    entities = set()

    def __new__(cls, vocabulary_file=GENES_VOCAB_FILE):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = read_file(vocabulary_file)
            cls._instance.prepare()
        return cls._instance

    def prepare(self):
        # GeneSymbol,GeneName,GeneID,AltGeneIDs,Synonyms,BioGRIDIDs,PharmGKBIDs,UniProtIDs
        for symbol, name, gid, alt_gids, synonyms, _, _, _ in self.data.itertuples(name=None, index=False):
            name = name.lower()
            synonyms = {s.lower() for s in str(synonyms).split("|") if s}
            synonyms.add(name)
            self.gene_id_to_synonyms[gid] = synonyms
            self.gene_id_to_alternatives[gid] = {s.lower() for s in str(alt_gids).split("|") if s}
            self.name_to_gene_id[name] = gid
            self.gene_id_to_name[gid] = name
            self.entities |= synonyms


class DrugbankVocabulary:
    _instance = None

    synonym_to_name = dict()
    name_to_synonyms = dict()
    entities = set()

    def __new__(cls, vocabulary_file=DRUGBANK_VOCAB_FILE):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = read_file(vocabulary_file)
            cls._instance.prepare()
        return cls._instance

    def prepare(self):
        # DrugBank ID,Accession Numbers,Common name,CAS,UNII,Synonyms,Standard InChI Key
        for _, _, name, _, _, synonyms, _ in self.data.itertuples(name=None, index=False):
            name = name.lower().strip()
            synonyms = {s.lower().strip() for s in str(synonyms).split("|") if s}
            # synonyms.add(name)
            for synonym in synonyms:
                self.synonym_to_name[synonym] = name
            self.name_to_synonyms[name] = synonyms
            self.entities |= synonyms
            self.entities.add(name)


if __name__ == "__main__":
    ChemicalVocab()
    DiseaseVocab()
    GeneVocab()
    DrugbankVocabulary()

    print(ChemicalVocab())
    print(list(ChemicalVocab().entities)[:10])

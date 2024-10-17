import csv
import glob
import os
import re
from collections import defaultdict
from typing import List, Tuple, Set, Optional, Dict
from xml.etree import ElementTree

from spacy.lang.en import English

from stp.config import CDR_PATH, CHEMPROT_PATH, DDI_PATH
from stp.data_generation.util import clean_entity_type

DDI_RANGE_INDICATORS = {"or", "and"}
DDI_RANGE_ABBREVIATION = re.compile(r"(.+)\s\(.*\)\s(.+)")

CONTENT_ID_TIT_ABS = re.compile(r"(\d+)\|t\|(.*?)\n\d+\|a\|(.*?)\n.*")
TAG_LINE_NORMAL = re.compile(r"(\d+)\t(\d+)\t(\d+)\t(.*?)\t(.*?)\t(.*?)\n")
RELATION_LINE_NORMAL = re.compile(r"\d+\tCID\t(.+)\t(.*)")

spacy_nlp = English()
spacy_nlp.add_pipe("sentencizer")


class TaggedEntity:

    def __init__(self, tag_tuple=None, doc_id: str = None, start: int = None, end: int = None, text=None,
                 ent_type=None, ent_id=None):
        if tag_tuple:
            doc_id, start, end, text, ent_type, ent_id = tag_tuple
        self.doc_id: str = doc_id
        self.start: int = int(start)
        self.end: int = int(end)
        self.text: str = text
        self.ent_type: str = ent_type
        self.ent_id: str = ent_id

    def __str__(self):
        return f"<Entity id={self.ent_id} [{self.start}-{self.end}] '{self.text}' type={self.ent_type}>"

    def __repr__(self):
        return self.__str__()


class Sentence:
    def __init__(self, doc_id, sent_id, text, start, end, relation_tags):
        self.doc_id = doc_id
        self.id = str(doc_id) + "-" + str(sent_id)
        self.text = text
        self.start = start
        self.end = end
        self.relation_tags: list = relation_tags

        self.relation_tags.sort(key=lambda t: t.start)

    def __str__(self):
        return f"<Sentence id={self.id} [{self.start}-{self.end}] '{self.text}'>"

    def __repr__(self):
        return self.__str__()

    def is_positive(self):
        """
        Checks if the sentence contains tags, which indicates a positive relation in between.
        :return:
        """
        return len(self.relation_tags) > 0


class Document:
    def __init__(self, doc_id: str, title: Optional[str], abstract: Optional[str],
                 tags: Optional[List[TaggedEntity]], relations: Optional[Set[Tuple]],
                 sentences: Optional[List[Sentence]], relation2subclass: Optional[Dict[Tuple, set]] = None):
        self.id: str = doc_id
        self.title: str = title
        self.abstract: str = abstract
        self.tags: List[TaggedEntity] = tags if tags is not None else list()
        self.relations: Set[Tuple] = relations if relations is not None else set()
        self.sentences: List[Sentence] = sentences if sentences is not None else list()
        self.relation2subclass = relation2subclass

        self.remove_duplicates_and_sort_tags()

        if not self.sentences:
            self._compute_nlp_indexes()

        self.relation_types: set = self.compute_relation_types()

    def compute_relation_types(self):
        relation_types = set()
        for e_id1, e_id2 in self.relations:
            e_id1_type = None
            e_id2_type = None

            for t in self.tags:
                if t.ent_id == e_id1:
                    e_id1_type = t.ent_type
                    break

            for t in self.tags:
                if t.ent_id == e_id2:
                    e_id2_type = t.ent_type
                    break

            if e_id1_type and e_id2_type:
                e_id1_type = clean_entity_type(e_id1_type)
                e_id2_type = clean_entity_type(e_id2_type)
                relation_types.add((e_id1_type, e_id2_type))
        return relation_types

    def get_sentences(self):
        return self.sentences

    def has_content(self):
        return True if (self.title or self.abstract) else False

    def _compute_nlp_indexes(self):
        self.sentences = list()

        if not self.has_content():
            return

        sentence_idx = 0
        # iterate over all text elements (title, abstract, sec1 title, sec1 text, sec2 title, ...)
        for text_element, offset in self.iterate_over_text_elements():
            doc_nlp = spacy_nlp(text_element)

            # iterate over sentences in each element
            for sent in doc_nlp.sents:
                sent_str = str(sent)
                start_pos = sent.start_char + offset
                end_pos = sent.end_char + offset

                # find all tags, that are part of the sentence
                possible_tags = {t for t in self.tags if t.start >= start_pos and t.end <= end_pos}

                for t in possible_tags:
                    # adjust the indices to the new relative boundaries of the sentence
                    t.start -= start_pos
                    t.end -= start_pos

                    assert t.text == sent_str[t.start:t.end], \
                        (f"Assertion failed for document {self.id=}\n"
                         f"\tin sentence {sentence_idx} '{sent_str}'\n"
                         f"\tfor {t=}\n"
                         f"\texpected '{t.text}'\n"
                         f"\tbut got  '{sent_str[t.start:t.end]}'")

                s = Sentence(doc_id=self.id, sent_id=sentence_idx, text=sent_str, start=start_pos, end=end_pos,
                             relation_tags=list(possible_tags))
                self.sentences.append(s)
                sentence_idx += 1

    def remove_duplicates_and_sort_tags(self):
        self.tags = list(set(self.tags))
        self.sort_tags()

    def sort_tags(self):
        try:
            self.tags = sorted(self.tags, key=lambda t: (t.start, t.end, t.ent_id))
        except TypeError:
            # No ent id given
            self.tags = sorted(self.tags, key=lambda t: (t.start, t.end))

    def get_text_content(self):
        return f"{self.title} {self.abstract}"

    def iterate_over_text_elements(self):
        running_offset = 0
        if self.title:
            yield self.title, 0
            running_offset += len(self.title) + 1
        if self.abstract:
            yield self.abstract, running_offset
            running_offset += len(self.abstract) + 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<Document {} {}>".format(self.id, self.title)


def load_from_pubtator(pubtator_content: str):
    match = CONTENT_ID_TIT_ABS.match(pubtator_content)
    if match:
        doc_id, title, abstract = match.group(1, 2, 3)
        title = title.strip()
        abstract = abstract.strip()
        doc_id = doc_id
    else:
        doc_id, title, abstract = None, None, None

    if pubtator_content:
        tags = [TaggedEntity(t) for t in TAG_LINE_NORMAL.findall(pubtator_content)]
        relations = {(e_id1, e_id2) for e_id1, e_id2 in RELATION_LINE_NORMAL.findall(pubtator_content)}
        if pubtator_has_composite_tags(tags):
            pubtator_split_composite_tags(tags)
    else:
        tags = []
        relations = []

    doc = Document(doc_id=doc_id, title=title, abstract=abstract, tags=tags, relations=relations, sentences=None)
    return doc


def pubtator_has_composite_tags(tags: [TaggedEntity]) -> bool:
    return '|' in str([''.join([t.ent_id for t in tags])])


def pubtator_split_composite_tags(tags: [TaggedEntity]) -> [TaggedEntity]:
    for t in tags:
        if '|' in t.ent_id:
            t.ent_id = t.ent_id.split("|")[0].strip()


def parse_cdr_run(run_name: str) -> List[Document]:
    """
    CDR runs have the PubTator format. Use the internal function of the Document class to parse the documents.
    :param run_name: name of the ChemicalDisease Run (CDR_DevelopmentSet.PubTator.txt, CDR_sample.txt,
    CDR_TestSet.PubTator.txt, CDR_TrainingSet.PubTator.txt)
    :return: list of sorted documents
    """
    with open(os.path.join(CDR_PATH, run_name), 'rt') as f:
        pub_strings = [s for s in f.read().split("\n\n") if len(s) > 0]

    docs = [load_from_pubtator(pub_str) for pub_str in pub_strings]
    docs.sort(key=lambda d: d.id)
    return docs


def parse_chemprot_run(run_name: str) -> List[Document]:
    """
    ChemProt runs are divided into 3 files each which contain title and abstract, entites and relations between them.
    UTF-8 encoding is required to parse special characters, which are used to express certain Gene names.
    :param run_name: name of the ChemicalProtein Benchmark Run (development, sample, test, training)
    :return: list of sorted documents
    """
    abstracts_path = os.path.join(CHEMPROT_PATH, f"chemprot_{run_name}_abstracts.tsv")
    entities_path = os.path.join(CHEMPROT_PATH, f"chemprot_{run_name}_entities.tsv")
    relations_path = os.path.join(CHEMPROT_PATH, f"chemprot_{run_name}_relations.tsv")
    if not os.path.exists(abstracts_path):
        # the test run has a different naming schema
        abstracts_path = os.path.join(CHEMPROT_PATH, f"chemprot_{run_name}_abstracts_gs.tsv")
        entities_path = os.path.join(CHEMPROT_PATH, f"chemprot_{run_name}_entities_gs.tsv")
        relations_path = os.path.join(CHEMPROT_PATH, f"chemprot_{run_name}_relations_gs.tsv")

    abstracts = dict()
    with open(abstracts_path, "rt", encoding="utf-8") as f:
        for doc_id, title, abstract in csv.reader(f, delimiter="\t"):
            abstracts[int(doc_id)] = (title, abstract)

    entities = dict()
    with open(entities_path, "rt", encoding="utf-8") as f:
        for doc_id, ent_id, ent_type, start, end, text in csv.reader(f, delimiter="\t"):
            doc_id = int(doc_id.strip())
            if doc_id not in entities:
                entities[doc_id] = list()
            entity = TaggedEntity(doc_id=str(doc_id), start=start, end=end, text=text, ent_type=ent_type, ent_id=ent_id)
            entities[doc_id].append(entity)

    relations = dict()
    relation2subclass = defaultdict(dict)
    with open(relations_path, "rt", encoding="utf-8") as f:
        for doc_id, relation_class, evaluate, relation, r1, r2 in csv.reader(f, delimiter="\t"):
            # skip relations that are ignored by evaluation
            if evaluate.strip() == "N":
                continue
            doc_id = int(doc_id.strip())
            if doc_id not in relations:
                relations[doc_id] = set()
            r1 = r1.split(":", 1)[-1] if r1.startswith("Arg1:") else r1
            r2 = r2.split(":", 1)[-1] if r2.startswith("Arg2:") else r2
            relations[doc_id].add((r1, r2))

            subclass = int(relation_class.split(":")[-1])
            if (r1, r2) not in relation2subclass[doc_id]:
                relation2subclass[doc_id][(r1, r2)] = set()
            relation2subclass[doc_id][(r1, r2)].add(subclass)

    docs = list()
    for doc_id, (title, abstract) in abstracts.items():
        doc_tags = entities.get(doc_id, list())
        doc_relations = relations.get(doc_id, set())
        doc_relation2subclass = relation2subclass.get(doc_id, set())
        doc = Document(doc_id=str(doc_id), title=title, abstract=abstract, tags=doc_tags, relations=doc_relations,
                       sentences=None, relation2subclass=doc_relation2subclass)
        docs.append(doc)
    docs.sort(key=lambda d: d.id)
    return docs


def apply_entity_index_range_rule(sentence: str, entity_text: str, entity_range: str):
    """
    This function tries to apply rules for the DrugDrugInteraction Benchmark, that potentially has relevant
    entities in some sentences, which are split up into multiple substrings.
    Both rules are based on the fact, that the entity_range is split up into two sub-strings corresponding  somehow.
    The first rule searches for abbreviations within in brackets. The second rule searches for logical connections
    expressed by the words 'and' or 'or'. If one of the rules can be applied, the entity_text is replaced with the
    range of the min and max value of the old entity_range. If none of the rules can be applied but the entity_range
    is split up, a ValueError is raised.
    :param sentence: current sentence
    :param entity_text: text content of the entity
    :param entity_range: range of the entity
    :return: entity_text, start, end
    """
    if ";" not in entity_range:
        start, end = entity_range.strip().split("-")
        return entity_text, int(start), int(end) + 1

    # get the first and all following; get the first and the last occurrence over all ranges
    range1, *range2 = entity_range.strip().split(";")
    start = int(range1.split("-")[0])
    end = int(range2[-1].split("-")[1]) + 1

    substring = sentence[start:end]

    if any(t in substring for t in DDI_RANGE_INDICATORS):
        # substring contains any of 'or', 'and', ...
        # e.g.: sentence: "nonheme and heme iron", tag: "nonheme iron", range: 83-89;100-103
        return substring, start, end
    if DDI_RANGE_ABBREVIATION.match(substring):
        # substring contains an abbreviation
        # e.g.: sentence: "heme (as CRBC) iron", tag: "heme iron", range: 142-145;157-160
        return substring, start, end

    raise ValueError(f"Could not find any allowed indicator in the part of the sentence to apply the rule\n"
                     f"\t{entity_range=} {entity_text=} {start=} {end=}\n"
                     f"\t{sentence[start:end]=}\n"
                     f"\t{sentence=}")


def parse_ddi_run(run_name: str) -> List[Document]:
    """
    DDI runs are stored per document in a file, formatted as XML. The document id is stored as the filename. Sentences
    contain entities and relations (pairs).
    :param run_name: name of the DrugDrugInteraction Benchmark run (Test, Train)
    :return: list of sorted documents
    """

    # use wildcard to get all documents of both sources MedLine and DrugBank
    run_dir = os.path.join(DDI_PATH, f"{run_name.title()}*", "*.xml")
    files = glob.glob(os.path.join(run_dir))
    docs = list()

    for file in files:
        root = ElementTree.parse(file).getroot()
        sent_offset = 0

        doc_id = root.get("id")

        sentences = list()
        doc_tags = list()
        doc_relations = list()

        for s in root.findall("sentence"):
            sent_relations = set()
            sent_tags = list()

            sent_id = s.get("id").strip()
            sent_text = s.get("text").strip()
            sent_end = sent_offset + len(sent_text)

            for relation in s.findall("pair"):
                if relation.get("ddi") == "false":
                    continue
                r1 = relation.get("e1").strip()
                r2 = relation.get("e2").strip()
                sent_relations.add(r1)
                sent_relations.add(r2)
                doc_relations.append((r1, r2))

            for entity in s.findall("entity"):
                ent_id = entity.get("id").strip()

                ent_range = entity.get("charOffset")
                ent_text = entity.get("text").strip()
                ent_type = entity.get("type").strip()

                ent_text, ent_start, ent_end = apply_entity_index_range_rule(sent_text, ent_text, ent_range)

                entity = TaggedEntity(doc_id=doc_id, start=ent_start, end=ent_end,
                                      text=ent_text, ent_type=ent_type, ent_id=ent_id)
                sent_tags.append(entity)
                doc_tags.append(entity)

            # filter relevant tags
            # relevant_tags = set()
            # for t1, t2 in itertools.combinations(sent_tags, 2):
            #     if (t1.ent_id, t2.ent_id) in doc_relations or (t2.ent_id, t1.ent_id) in doc_relations:
            #         relevant_tags.add(t1)
            #         relevant_tags.add(t2)

            for t in sent_tags:
                assert t.text == sent_text[t.start:t.end], \
                    (f"Assertion failed for document {doc_id=}\n"
                     f"\tin sentence {sent_id} '{sent_text}'\n"
                     f"\tfor {t=}\n"
                     f"\texpected '{t.text}'\n"
                     f"\tbut got  '{sent_text[t.start:t.end]}'")

            sentence = Sentence(doc_id=doc_id, sent_id=sent_id, text=sent_text, start=sent_offset,
                                end=sent_end, relation_tags=sent_tags)
            sentences.append(sentence)
            sent_offset = sent_end + 1

        title = sentences[0].text
        abstract = " ".join(s.text for s in sentences[1:])

        doc = Document(doc_id=doc_id, title=title, abstract=abstract, tags=doc_tags, relations=set(doc_relations),
                       sentences=sentences)
        docs.append(doc)

    docs.sort(key=lambda d: d.id)
    return docs

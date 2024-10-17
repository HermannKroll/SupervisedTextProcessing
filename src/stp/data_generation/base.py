import ast
from abc import ABC, abstractmethod

import pandas as pd
from snorkel.augmentation import transformation_function, ApplyAllPolicy, PandasTFApplier


class DataGeneration(ABC):
    def __init__(self, data_source, use_obfuscation: bool, verbose: bool = True):
        self.use_obfuscation = use_obfuscation
        self.data_source = data_source
        self.verbose = verbose
        self._generate_data()

    @abstractmethod
    def _generate_data(self):
        self.sentences = pd.DataFrame()
        raise NotImplementedError("Implement this method")

    def _apply_transformation_policies(self):
        if self.use_obfuscation:
            transformation_functions = [self.obfuscate_entities]

            policy = ApplyAllPolicy(len(transformation_functions), keep_original=False)
            applier = PandasTFApplier(transformation_functions, policy)
            self.sentences = pd.DataFrame(applier.apply(df=self.sentences, progress_bar=self.verbose))

    @staticmethod
    @transformation_function()
    def obfuscate_entities(row: pd.Series):
        sentence: str = row.sentence
        i = 1
        if "subjects" in row:
            if isinstance(row.subjects, str):
                subjects = ast.literal_eval(row.subjects)
            elif isinstance(row.subjects, list):
                subjects = row.subjects
            else:
                raise NotImplementedError("Unknown subject type")
            for subject in subjects:
                sentence = sentence.replace(subject, f'<entity{i}>')
                i += 1
        if "objects" in row:
            if isinstance(row.objects, str):
                objects = ast.literal_eval(row.objects)
            elif isinstance(row.subjects, list):
                objects = row.objects
            else:
                raise NotImplementedError("Unknown subject type")
            for object in objects:
                sentence = sentence.replace(object, f'<entity{i}>')
                i += 1
        if "entities" in row:
            for i, (entity, entity_type, *_) in enumerate(row.entities, start=1):
                sentence = sentence.replace(entity, f'<entity{i}>')
        row.sentence = sentence
        return row

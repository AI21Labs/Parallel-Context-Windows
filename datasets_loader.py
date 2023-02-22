import logging
from abc import ABC
from typing import Dict, Optional

import pandas as pd
from datasets import load_dataset

from constants import PROMPTS


UTTERANCE_PREFIX = 'utterance: '

INTENT_PREFIX = 'intent: '

LABEL_TOKENS = 'label_tokens'

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class ClassificationDatasetAccess(ABC):
    name: str
    dataset: Optional[str] = None
    subset: Optional[str] = None
    x_column: str = 'text'
    y_label: str = 'label'
    x_prefix: str = "Review: "
    y_prefix: str = "Sentiment: "
    label_mapping: Optional[Dict] = None
    map_labels: bool = True

    def __init__(self):
        super().__init__()
        if self.dataset is None:
            self.dataset = self.name
        train_dataset, test_dataset = self._load_dataset()
        train_df = train_dataset.to_pandas()
        test_df = test_dataset.to_pandas()
        _logger.info(f"loaded {len(train_df)} training samples & {len(test_df)} test samples")

        if self.map_labels:
            hf_default_labels = train_dataset.features[self.y_label]
            default_label_mapping = dict(enumerate(hf_default_labels.names)) if hasattr(
                train_dataset.features[self.y_label], 'names') else None
            self._initialize_label_mapping(default_label_mapping)

        self.train_df = self.apply_format(train_df)
        self.test_df = self.apply_format(test_df, test=True)

    def _initialize_label_mapping(self, default_label_mapping):
        if self.label_mapping:
            _logger.info("overriding default label mapping")
            if default_label_mapping:
                _logger.info([f"{default_label_mapping[k]} -> "
                              f"{self.label_mapping[k]}" for k in self.label_mapping.keys()])
        else:
            _logger.info(f"using default label mapping: {default_label_mapping}")
            self.label_mapping = default_label_mapping

    def _load_dataset(self):
        if self.subset is not None:
            dataset = load_dataset(self.dataset, self.subset)
        else:
            dataset = load_dataset(self.dataset)
        if 'validation' in dataset:
            return dataset['train'], dataset['validation']
        if 'test' not in dataset:
            _logger.info("no test or validation found, splitting train set instead")
            dataset = dataset['train'].train_test_split(seed=42)

        return dataset['train'], dataset['test']

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_y_token_labels(self, df, test):
        if self.map_labels:
            df[LABEL_TOKENS] = df[self.y_label].map(self.label_mapping)
        else:
            df[LABEL_TOKENS] = df[self.y_label]
        return df

    @property
    def labels(self):
        if self.map_labels:
            return self.label_mapping.values()
        else:
            return self.test_df[LABEL_TOKENS].unique()

    def apply_format(self, df, test=False):
        df = self.generate_x_text(df)
        df = self.generate_y_token_labels(df, test)
        if test:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}".rstrip(), axis=1)
        else:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}",
                                   axis=1)
        return df


class SST5(ClassificationDatasetAccess):
    name = 'sst5'
    dataset = 'SetFit/sst5'
    label_mapping = {0: 'terrible', 1: 'bad', 2: 'okay', 3: 'good', 4: 'great'}


class RTE(ClassificationDatasetAccess):
    name = 'rte'
    dataset = 'super_glue'
    subset = 'rte'
    x_prefix = ''
    y_prefix = 'prediction: '
    label_mapping = {0: 'True', 1: 'False'}

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df.apply(lambda x: f"premise: {x['premise']}\nhypothesis: {x['hypothesis']}", axis=1)
        return df


class CB(RTE):
    name = 'cb'
    subset = 'cb'
    label_mapping = {0: 'true', 1: 'false', 2: 'neither'}


class SUBJ(ClassificationDatasetAccess):
    name = 'subj'
    dataset = 'SetFit/subj'
    label_mapping = {0: 'objective', 1: 'subjective'}
    x_prefix = 'Input: '
    y_prefix = 'Type: '


class CR(ClassificationDatasetAccess):
    name = 'cr'
    dataset = 'SetFit/CR'
    label_mapping = {0: 'negative', 1: 'positive'}


class AGNEWS(ClassificationDatasetAccess):
    name = 'agnews'
    dataset = 'ag_news'
    label_mapping = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
    x_prefix = 'input: '
    y_prefix = 'type: '


class DBPEDIA(ClassificationDatasetAccess):
    name = 'dbpedia'
    dataset = 'dbpedia_14'
    label_mapping = {0: 'company',
                     1: 'school',
                     2: 'artist',
                     3: 'athlete',
                     4: 'politics',
                     5: 'transportation',
                     6: 'building',
                     7: 'nature',
                     8: 'village',
                     9: 'animal',
                     10: 'plant',
                     11: 'album',
                     12: 'film',
                     13: 'book'}
    x_prefix = 'input: '
    y_prefix = 'type: '

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['content']
        return df


class SST2(ClassificationDatasetAccess):
    name = 'sst2'

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['sentence']
        return df


class TREC(ClassificationDatasetAccess):
    name = 'trec'
    y_label = 'coarse_label'
    x_prefix = "Question: "
    y_prefix = "Type: "
    label_mapping = {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: 'numeric'}


class TRECFINE(ClassificationDatasetAccess):
    name = 'trecfine'
    dataset = 'trec'
    y_label = 'fine_label'
    x_prefix = "Question: "
    y_prefix = "Type: "
    # labels mapping based on: https://aclanthology.org/C16-1116.pdf, https://aclanthology.org/C02-1150.pdf
    label_mapping = {0: 'abbreviation abbreviation',
                     1: 'abbreviation expansion',
                     2: 'entity animal',
                     3: 'entity body',
                     4: 'entity color',
                     5: 'entity creation',
                     6: 'entity currency',
                     7: 'entity disease',
                     8: 'entity event',
                     9: 'entity food',
                     10: 'entity instrument',
                     11: 'entity language',
                     12: 'entity letter',
                     13: 'entity other',
                     14: 'entity plant',
                     15: 'entity product',
                     16: 'entity religion',
                     17: 'entity sport',
                     18: 'entity substance',
                     19: 'entity symbol',
                     20: 'entity technique',
                     21: 'entity term',
                     22: 'entity vehicle',
                     23: 'entity word',
                     24: 'description definition',
                     25: 'description description',
                     26: 'description manner',
                     27: 'description reason',
                     28: 'human group',
                     29: 'human individual',
                     30: 'human title',
                     31: 'human description',
                     32: 'location city',
                     33: 'location country',
                     34: 'location mountain',
                     35: 'location other',
                     36: 'location state',
                     37: 'numeric code',
                     38: 'numeric count',
                     39: 'numeric date',
                     40: 'numeric distance',
                     41: 'numeric money',
                     42: 'numeric order',
                     43: 'numeric other',
                     44: 'numeric period',
                     45: 'numeric percent',
                     46: 'numeric speed',
                     47: 'numeric temperature',
                     48: 'numeric size',
                     49: 'numeric weight'}


class YELP(ClassificationDatasetAccess):
    name = 'yelp'
    dataset = 'yelp_review_full'
    x_prefix = 'review: '
    y_prefix = 'stars: '
    label_mapping = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}


class BANKING77(ClassificationDatasetAccess):
    name = 'banking77'
    x_prefix = 'query: '
    y_prefix = INTENT_PREFIX

    def _initialize_label_mapping(self, default_label_mapping):
        default_label_mapping = {k: v.replace('_', ' ') for k, v in default_label_mapping.items()}
        super()._initialize_label_mapping(default_label_mapping)


class NLU(ClassificationDatasetAccess):
    name = 'nlu'
    dataset = 'nlu_evaluation_data'
    x_prefix = UTTERANCE_PREFIX
    y_prefix = INTENT_PREFIX
    label_mapping = {0: 'alarm query', 1: 'alarm remove', 2: 'alarm set', 3: 'audio volume down',
                     4: 'audio volume mute', 5: 'audio volume other', 6: 'audio volume up', 7: 'calendar query',
                     8: 'calendar remove', 9: 'calendar set', 10: 'cooking query', 11: 'cooking recipe',
                     12: 'datetime convert', 13: 'datetime query', 14: 'email add contact', 15: 'email query',
                     16: 'email query contact', 17: 'email sendemail', 18: 'general affirm', 19: 'general command stop',
                     20: 'general confirm', 21: 'general dont care', 22: 'general explain', 23: 'general greet',
                     24: 'general joke', 25: 'general negate', 26: 'general praise', 27: 'general quirky',
                     28: 'general repeat', 29: 'iot cleaning', 30: 'iot coffee', 31: 'iot hue light change',
                     32: 'iot hue light dim', 33: 'iot hue light off', 34: 'iot hue lighton', 35: 'iot hue light up',
                     36: 'iot wemo off', 37: 'iot wemo on', 38: 'lists create or add', 39: 'lists query',
                     40: 'lists remove', 41: 'music dislikeness', 42: 'music likeness', 43: 'music query',
                     44: 'music settings', 45: 'news query', 46: 'play audiobook', 47: 'play game', 48: 'play music',
                     49: 'play podcasts', 50: 'play radio', 51: 'qa currency', 52: 'qa definition', 53: 'qa factoid',
                     54: 'qa maths', 55: 'qa stock', 56: 'recommendation events', 57: 'recommendation locations',
                     58: 'recommendation movies', 59: 'social post', 60: 'social query', 61: 'takeaway order',
                     62: 'takeaway query', 63: 'transport query', 64: 'transport taxi', 65: 'transport ticket',
                     66: 'transport traffic', 67: 'weather query'}


class NLUSCENARIO(ClassificationDatasetAccess):
    name = 'nluscenario'
    dataset = 'nlu_evaluation_data'
    x_prefix = UTTERANCE_PREFIX
    y_prefix = 'scenario: '
    y_label = 'scenario'
    map_labels = False


class CLINIC150(BANKING77):
    name = "clinic150"
    dataset = 'clinc_oos'
    subset = 'plus'
    y_label = "intent"
    x_prefix = UTTERANCE_PREFIX
    y_prefix = INTENT_PREFIX


DATASET_NAMES2LOADERS = {'sst5': SST5, 'sst2': SST2, 'agnews': AGNEWS, 'dbpedia': DBPEDIA, 'trec': TREC, 'cr': CR,
                         'cb': CB, 'rte': RTE, 'subj': SUBJ, 'yelp': YELP, 'banking77': BANKING77,
                         'nlu': NLU, 'nluscenario': NLUSCENARIO, 'trecfine': TRECFINE,
                         'clinic150': CLINIC150}

if __name__ == '__main__':
    for ds_name, da in DATASET_NAMES2LOADERS.items():
        _logger.info(ds_name)
        _logger.info(da().train_df[PROMPTS].iloc[0])

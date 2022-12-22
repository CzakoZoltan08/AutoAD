from enum import Enum


class NoiseType(Enum):
    NONE = 0
    DUPLICATES = 1
    IRRELEVANT_FEATURES = 2
    LABEL_ERROR = 3

from enum import Enum


class NoiseType(Enum):
    NONE = "NONE"
    DUPLICATES = "DUPLICATES"
    IRRELEVANT_FEATURES = "IRRELEVANT_FEATURES"
    LABEL_ERROR = "LABEL_ERROR"

from enum import Enum


class AnomalyType(Enum):
    LOCAL = 1
    GLOBAL = 2
    CLUSTER = 3
    CONTEXTUAL = 4

from enum import Enum


class AnomalyType(Enum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"
    CLUSTER = "CLUSTER"
    CONTEXTUAL = "CONTEXTUAL"

from enum import Enum


class Dataset(str, Enum):
    ALOI = "1_ALOI"
    ANNTHYROID = "2_annthyroid"
    BACKDOOR = "3_backdoor"
    BREASTW = "4_breastw"
    CAMPAING = "5_campaign"
    CARDIO = "6_cardio"

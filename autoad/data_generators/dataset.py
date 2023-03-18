from enum import Enum


class Dataset(str, Enum):
    WINE = "45_wine"
    LYMPHOGRAPHY = "21_Lymphography"
    GLASS = "14_glass"
    STAMPS = "37_Stamps"
    IONOSPHERE = "18_Ionosphere"
    BREASTW = "4_breastw"
    YEAST = "47_yeast"
    PAGEBLOCKS = "27_PageBlocks"
    ANNTHYROID = "2_annthyroid"
    CARDIO = "6_cardio"

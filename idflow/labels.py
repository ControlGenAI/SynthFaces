"""Label vocabularies (FairFace categories) and group enumeration.

Label order everywhere in this repo is ``[age, race, gender]``.
"""
from itertools import product

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]
GENDER_LABELS = ["Male", "Female"]

LABEL_NAMES = ["age", "race", "gender"]
CLASS_NUMS = [len(AGE_LABELS), len(RACE_LABELS), len(GENDER_LABELS)]  # [9, 7, 2]

N_GROUPS = CLASS_NUMS[0] * CLASS_NUMS[1] * CLASS_NUMS[2]  # 126


def all_groups():
    """All (age, race, gender) index combinations, in a fixed order."""
    return list(product(range(CLASS_NUMS[0]), range(CLASS_NUMS[1]), range(CLASS_NUMS[2])))

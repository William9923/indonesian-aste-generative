import os


class Path:
    RESOURCES = "resources"
    DATA = "data"
    PRETRAINED = "pretrained"
    MODEL = "bin"
    REPORT = "reports"

    FILTERED = "filter"
    UNFILTERED = "unfilter"

    TRAIN = "train.txt"
    TEST = "test.txt"
    VAL = "dev.txt"

    TRAIN_FILTERED_PATH = os.path.join(DATA, FILTERED, TRAIN)
    TEST_FILTERED_PATH = os.path.join(DATA, FILTERED, TEST)
    VAL_FILTERED_PATH = os.path.join(DATA, FILTERED, VAL)

    TRAIN_UNFILTERED_PATH = os.path.join(DATA, UNFILTERED, TRAIN)
    TEST_UNFILTERED_PATH = os.path.join(DATA, UNFILTERED, TEST)
    VAL_UNFILTERED_PATH = os.path.join(DATA, UNFILTERED, VAL)

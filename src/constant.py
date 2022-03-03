import os


class ModelType:
    T5Model = "t5"


class ProcessType:
    DoTrain = "train"
    DoTest = "test"


class DatasetType:
    FILTERED = "filter"
    UNFILTERED = "unfilter"


class PostprocessType:
    EDITDISTANCE = "edit-distance"
    EMBEDDING = "embedding"


class Path:
    RESOURCES = "resources"
    DATA = "data"
    PRETRAINED = "pretrained"
    MODEL = "bin"
    REPORT = "reports"

    TRAIN = "train.txt"
    TEST = "test.txt"
    VAL = "dev.txt"

    TRAIN_FILTERED_PATH = os.path.join(DATA, DatasetType.FILTERED, TRAIN)
    TEST_FILTERED_PATH = os.path.join(DATA, DatasetType.FILTERED, TEST)
    VAL_FILTERED_PATH = os.path.join(DATA, DatasetType.FILTERED, VAL)

    TRAIN_UNFILTERED_PATH = os.path.join(DATA, DatasetType.UNFILTERED, TRAIN)
    TEST_UNFILTERED_PATH = os.path.join(DATA, DatasetType.UNFILTERED, TEST)
    VAL_UNFILTERED_PATH = os.path.join(DATA, DatasetType.UNFILTERED, VAL)

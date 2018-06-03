import json_serializable
from enum import Enum


class Mode(Enum):
    TRAIN = 0
    TEST = 1
    MEASURE = 2
    GENERATE_MODELS = 3


class Config(json_serializable.JsonSerializable):

    def __init__(self):
        self.TRAINING_PATTERN_LENGTH = 50
        self.TRAINING_EPOCHS = 60
        self.TRAINING_BATCH_SIZE = 16
        self.HIDDEN_LAYER_SIZE = 256
        self.DROPOUT = 0.2
        self.ACTIVATION_FUNC = "tanh"
        self.LOSS_FUNC = "categorical_crossentropy"
        self.OPTIMIZER = "rmsprop"

        self.TEST_STARTING_INDEX = 0
        self.TEST_PATTERN_LENGTH = 400

        self.GPUS = 1
        self.CPUS = 6
        self.USE_GPU = True

        self.TRAINING_PATH = "models/"
        self.DATASET_PATH = "midi_songs/**/*.mid"
        self.OUTPUT_PATH = "output/"

        self.MODEL_NAME = "2018-05-08_23-38"
        self.MODEL_PATH = self.TRAINING_PATH + self.MODEL_NAME
        self.MODEL_INFO_PATH = self.MODEL_PATH + "_ModelInfo"

        self.EXEC_MODE = Mode.TEST

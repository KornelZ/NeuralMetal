import json_serializable


class Config(json_serializable.JsonSerializable):

    def __init__(self):
        self.TRAINING_PATTERN_LENGTH = 50
        self.TRAINING_EPOCHS = 60
        self.TRAINING_BATCH_SIZE = 16
        self.HIDDEN_LAYER_SIZE_MULTIPLIER = 2
        self.DROPOUT = 0.2
        self.ACTIVATION_FUNC = "sigmoid"
        self.LOSS_FUNC = "categorical_crossentropy"
        self.OPTIMIZER = "rmsprop"

        self.TEST_STARTING_INDEX = 0
        self.TEST_PATTERN_LENGTH = 400
        self.NOTE_OFFSET = 0.5

        self.GPUS = 1
        self.CPUS = 6
        self.USE_GPU = True

        self.TRAINING_PATH = "models/"
        self.DATASET_PATH = "midi_songs/**/*.mid"
        self.OUTPUT_PATH = "output/"

        self.IS_TRAINING = False
        self.MODEL_PATH = self.TRAINING_PATH + "2018-05-08_21-12"
        self.MODEL_INFO_PATH = self.MODEL_PATH + "_ModelInfo"

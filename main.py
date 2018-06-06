import numpy as np
from model import Model, ModelInfo
from preprocess import get_notes, prepare_input, parse_song
from config import Config, Mode
import datetime
from music21 import note, instrument, stream, chord
import reportgenerator
import sys

def save_output(output, song, path):
    midi_stream = stream.Stream(output)
    path = path + song[10:]
    path = path.replace("\\", "_")
    midi_stream.write('midi', fp=path)

"""Transforms of list of strings representing the notes into a midi stream"""
def to_file(output):
    offset = 0
    out_notes = []

    for pattern in output:
        pattern_and_offset = pattern.split('_')
        pattern = pattern_and_offset[0]
        try:
            offset_step = float(pattern_and_offset[1])
        except:
            offset_step = 0
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for n in notes_in_chord:
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.AcousticGuitar()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            out_notes.append(new_chord)
        else:
            if pattern == "-1":
                new_note = note.Rest()
            else:
                new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.AcousticGuitar()
            out_notes.append(new_note)

        offset += offset_step
    return out_notes

"""Returns generated song from source data fragment"""
def predict(data, pitches, num_unique_notes, model, config):
    #create dictionary of notes' indices and note values from given array of pitches
    int_to_note = dict((num, note) for num, note in enumerate(pitches))
    pattern = data[config.TEST_STARTING_INDEX]
    output = []

    for note_index in range(config.TEST_PATTERN_LENGTH):
        pred_in = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(pred_in)
        #choose note with highest probability
        index = np.argmax(prediction)
        result = int_to_note[index]
        output.append(result)
        #append new note to pattern, normalize it and shift one to the right
        pattern = np.append(pattern, [[index / float(num_unique_notes)]], axis=0)
        pattern = pattern[1:len(pattern)]

    print(output)
    return output

"""Generates new songs of based on model and source song, saves fragment of source song(sample) and output song"""
def test(model_info, config):
    model = Model(config.USE_GPU, config.GPUS, config.CPUS)
    model.load(config.MODEL_PATH)
    for song in model_info.songs:
        sample, notes = parse_song(song, config)
        data, _, _ = prepare_input(notes, model_info.num_unique_notes, config.TRAINING_PATTERN_LENGTH)
        output = predict(data, model_info.pitches, model_info.num_unique_notes, model, config)
        save_output(to_file(output), song, config.OUTPUT_PATH)
        save_output(sample, song, config.SAMPLES_PATH)

"""Creates data table of sequences of specifies length, target labels and info about model"""
def preprocess(sequence_length, config):
    notes, songs = get_notes(config)
    num_unique_notes = len(set(notes))
    data, labels, pitches = prepare_input(notes, num_unique_notes, sequence_length)

    return data, labels, ModelInfo(pitches, num_unique_notes, songs)


def get_training_path(config):
    now = datetime.datetime.now()
    return config.TRAINING_PATH + now.strftime("%Y-%m-%d_%H-%M")

"""Trains the model on songs from midi_songs and saves it and its config"""
def train(config):
    data, labels, model_info = preprocess(config.TRAINING_PATTERN_LENGTH, config)
    model = Model(config.USE_GPU, config.GPUS, config.CPUS)
    model.init_model(data.shape, model_info.num_unique_notes, config.HIDDEN_LAYER_SIZE,
                     config.DROPOUT, config.ACTIVATION_FUNC, config.LOSS_FUNC, config.OPTIMIZER)
    model.train(data, labels, epochs=config.TRAINING_EPOCHS, batch_size=config.TRAINING_BATCH_SIZE)
    path = get_training_path(config)
    model_info.serialize(path)
    model.save(path)
    config.serialize(path)

"""Runs sequence mining algorithms on source songs and generated songs, and creates a report"""
def measure(config):
    report = reportgenerator.Report()
    report.generate_report(config)


def generate_models():
    c = Config()
    activations = ["softmax"]
    pattern_lens = [25, 50, 75, 100]
    hidden_layer_sizes = [64, 128, 256]
    epochs = [60, 120, 180]

    for ep in epochs:
        for act in activations:
            for hidden in hidden_layer_sizes:
                for lens in pattern_lens:
                    print("epochs: {0}, pattern length: {1}, activation: {2}, hidden layer: {3}"
                          .format(ep, lens, act, hidden))
                    c.ACTIVATION_FUNC = act
                    c.TRAINING_PATTERN_LENGTH = lens
                    c.HIDDEN_LAYER_SIZE = hidden
                    c.TRAINING_EPOCHS = ep
                    train(c)


def main():
    config = Config()
    if len(sys.argv) >= 2:
        config.MODEL_NAME = sys.argv[1]
    if len(sys.argv) >= 3:
        if sys.argv[2] == "train":
            config.EXEC_MODE = Mode.TRAIN
        elif sys.argv[2] == "test":
            config.EXEC_MODE = Mode.TEST
        elif sys.argv[2] == "measure":
            config.EXEC_MODE = Mode.MEASURE
        elif sys.argv[2] == "generate":
            config.EXEC_MODE = Mode.GENERATE_MODELS
    if config.EXEC_MODE == Mode.TRAIN:
        train(config)
    elif config.EXEC_MODE == Mode.TEST:
        model_info = ModelInfo.deserialize(config.MODEL_INFO_PATH)
        c = Config.deserialize(config.CONFIG_INFO_PATH)
        config.TRAINING_PATTERN_LENGTH = c.TRAINING_PATTERN_LENGTH
        test(model_info, config)
    elif config.EXEC_MODE == Mode.MEASURE:
        measure(config)
    elif config.EXEC_MODE == Mode.GENERATE_MODELS:
        generate_models()


#Przykladowa egzekucja programu:
#trening: python main.py _ train
#generacja utworów: python main.py 2018-06-02_16-24 test
#generacja raportów: python main.py 2018-06-02_16-24 measure
#generacja wielu modeli: python main.py _ generate
#pierwszy argument jest nazwą modelu z folderu models i jest wymagany dla komend test i measure
#w przypadku komend train i generate należy podać _(lub inny niepusty string)
# wywołanie: python main.py bez argumentów wywoła program zgodnie z konfiguracją w config.py

if __name__ == "__main__":
    main()

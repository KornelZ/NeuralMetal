import numpy as np
from model import Model, ModelInfo
from preprocess import get_notes, prepare_input, parse_song
from config import Config
import datetime
from music21 import note, instrument, stream, chord
import reportgenerator


def to_file(output, song, config):
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

    midi_stream = stream.Stream(out_notes)
    path = config.OUTPUT_PATH + song[10:]
    path = path.replace("\\", "_")
    midi_stream.write('midi', fp=path)


def predict(data, pitches, num_unique_notes, model, config):
    int_to_note = dict((num, note) for num, note in enumerate(pitches))
    pattern = data[config.TEST_STARTING_INDEX]
    output = []

    for note_index in range(config.TEST_PATTERN_LENGTH):
        pred_in = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(pred_in)

        index = np.argmax(prediction)
        result = int_to_note[index]
        output.append(result)
        pattern = np.append(pattern, [[index / float(num_unique_notes)]], axis=0)
        pattern = pattern[1:len(pattern)]

    print(output)
    return output


def test(model_info, config):
    model = Model(config.USE_GPU, config.GPUS, config.CPUS)
    model.load(config.MODEL_PATH)
    for song in model_info.songs:
        n = []
        parse_song(song, n)
        data, _, _ = prepare_input(n, model_info.num_unique_notes, config.TRAINING_PATTERN_LENGTH)
        output = predict(data, model_info.pitches, model_info.num_unique_notes, model, config)
        to_file(output, song, config)


def preprocess(sequence_length, config):
    notes, songs = get_notes(config)
    num_unique_notes = len(set(notes))
    data, labels, pitches = prepare_input(notes, num_unique_notes, sequence_length)

    return data, labels, ModelInfo(pitches, num_unique_notes, songs)


def get_training_path(config):
    now = datetime.datetime.now()
    return config.TRAINING_PATH + now.strftime("%Y-%m-%d_%H-%M")


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

def measure(config):
    report = reportgenerator.Report()
    report.generate_report(config)


def generate_models(config):
    c = Config()
    activations = ["softmax"]
    pattern_lens = [100, 200, 400]
    hidden_layer_sizes = [128, 256]
    epochs = [60, 120, 180]

    for ep in epochs:
        for act in activations:
            for hidden in hidden_layer_sizes:
                c.ACTIVATION_FUNC = act
                c.HIDDEN_LAYER_SIZE = hidden
                c.TRAINING_EPOCHS = ep
                train(c)


def main():
    config = Config()
    if config.IS_GENERATING:
        generate_models(config)
    elif config.IS_MEASURE:
        measure(config)
    else:
        if config.IS_TRAINING:
            train(config)
        else:
            model_info = ModelInfo.deserialize(config.MODEL_INFO_PATH)
            test(model_info, config)


if __name__ == "__main__":
    main()

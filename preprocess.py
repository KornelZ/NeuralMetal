import glob
import numpy as np
from music21 import converter, instrument, note, chord
from keras.utils import to_categorical
from config import Config


def parse_song(file, notes):
    try:
        m = converter.parse(file)
        parts = instrument.partitionByInstrument(m)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = m.flat.notesAndRests

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                print(element.pitch)
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                print(element)
            elif isinstance(element, note.Rest):
                notes.append(str(-1))
                print("Rest")
    except:
        print("error")


def get_notes(config):
    notes = []
    songs = []
    for file in glob.glob(config.DATASET_PATH, recursive=True):
        print(file)
        parse_song(file, notes)
        songs.append(file)

    return notes, songs


def prepare_input(notes, num_unique_notes, sequence_length):
    pitch_names = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    data = []
    labels = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        data.append([note_to_int[c] for c in seq_in])
        labels.append([note_to_int[seq_out]])

    data, labels = normalize(data, labels, sequence_length, num_unique_notes)

    return data, labels, pitch_names


def normalize(data, labels, sequence_length, num_unique_notes):
    n_patterns = len(data)
    data = np.reshape(data, (n_patterns, sequence_length, 1))
    data = data / float(num_unique_notes)
    labels = to_categorical(labels)
    return data, labels




import glob
import numpy as np
from music21 import converter, instrument, note, chord
from keras.utils import to_categorical
from offset import limit_offset

"""Parses song creating sample source fragment and song consisting of notes, chords and breaks in music"""
def parse_song(file, config):
    sample = []
    notes = []
    try:
        m = converter.parse(file)
        parts = instrument.partitionByInstrument(m)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = m.flat.notesAndRests
        prev_offset = 0
        for element in notes_to_parse:
            offset = "_" + str(limit_offset(round(element.offset - prev_offset, 2)))
            prev_offset = element.offset
            print(offset)
            if isinstance(element, note.Note):
                sample.append(element)
                notes.append(str(element.pitch) + offset)
                print(element.pitch)
            elif isinstance(element, chord.Chord):
                sample.append(element)
                notes.append('.'.join(str(n) for n in element.normalOrder) + offset)
                print(element)
            elif isinstance(element, note.Rest):
                sample.append(element)
                notes.append(str(-1) + offset)
                print("Rest")
    except:
        print("error")
    if len(sample) >= config.TRAINING_PATTERN_LENGTH:
        return sample[:config.TRAINING_PATTERN_LENGTH], notes
    else:
        return None, notes

"""Parses all songs in dataset path and concatenates their notes"""
def get_notes(config):
    notes = []
    songs = []
    for file in glob.glob(config.DATASET_PATH, recursive=True):
        print(file)
        _, note_list = parse_song(file, config)
        notes += note_list
        songs.append(file)

    return notes, songs

"""Returns prepared data table, target labels and sorted list of unique notes"""
def prepare_input(notes, num_unique_notes, sequence_length):
    pitch_names = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    data = []
    labels = []
    #input sequence is x notes and target is x + 1 note
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        data.append([note_to_int[c] for c in seq_in])
        labels.append([note_to_int[seq_out]])

    data, labels = normalize(data, labels, sequence_length, num_unique_notes)

    return data, labels, pitch_names

"""Normalizes data values to [0, 1] range, resizes array to 2d arr, sets labels to one hot encoding"""
def normalize(data, labels, sequence_length, num_unique_notes):
    n_patterns = len(data)
    data = np.reshape(data, (n_patterns, sequence_length, 1))
    data = data / float(num_unique_notes)
    labels = to_categorical(labels)
    return data, labels




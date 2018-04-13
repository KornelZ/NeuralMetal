import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.utils import to_categorical


def get_notes():
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        m = converter.parse(file)
        parts = instrument.partitionByInstrument(m)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = m.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                print(element.pitch)
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                print(element)
    to_file(notes)
    return notes


def prepare_input(notes, num_unique_notes, sequence_length):
    pitch_names = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    data = []
    labels = []

    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        data.append([note_to_int[c] for c in seq_in])
        labels.append([note_to_int[seq_out]])

    n_patterns = len(data)
    data = np.reshape(data, (n_patterns, sequence_length, 1))
    data = data / float(num_unique_notes)
    labels = to_categorical(labels)

    return data, labels, pitch_names


def to_file(output):
    offset = 0
    out_notes = []

    for pattern in output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for n in notes_in_chord:
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.Guitar()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            out_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Guitar()
            out_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(out_notes)
    midi_stream.write('midi', fp="output.mid")
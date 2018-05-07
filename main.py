import numpy as np
from model import Model
from preprocess import get_notes, prepare_input, to_file
import config


def predict(data, pitches, num_unique_notes, pattern_length, model):
    int_to_note = dict((num, note) for num, note in enumerate(pitches))
    pattern = data[config.TEST_STARTING_INDEX]
    output = []

    for note_index in range(pattern_length):
        pred_in = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(pred_in)

        index = np.argmax(prediction)
        result = int_to_note[index]
        output.append(result)
        pattern = np.append(pattern, [[index / float(num_unique_notes)]], axis=0)
        pattern = pattern[1:len(pattern)]

    print(output)
    return output


def preprocess(sequence_length):
    notes = get_notes()
    num_unique_notes = len(set(notes))
    data, labels, pitches = prepare_input(notes, num_unique_notes, sequence_length)

    return data, labels, pitches, num_unique_notes


def main():
    data, labels, pitches, num_unique_notes = preprocess(sequence_length=config.TRAINING_PATTERN_LENGTH)
    model = Model(config.USE_GPU, config.GPUS, config.CPUS)
    model.init_model(data.shape, num_unique_notes, num_unique_notes * config.HIDDEN_LAYER_SIZE_MULTIPLIER,
                     config.DROPOUT, config.ACTIVATION_FUNC, config.LOSS_FUNC, config.OPTIMIZER)
    model.train(data, labels, epochs=config.TRAINING_EPOCHS, batch_size=config.TRAINING_BATCH_SIZE)
    result = predict(data, pitches, num_unique_notes, config.TEST_PATTERN_LENGTH, model)
    to_file(result)


if __name__ == "__main__":
    main()

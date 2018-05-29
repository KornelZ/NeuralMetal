from keras.layers import Dense, Dropout, LSTM, Activation
from keras import Sequential, backend
from keras.models import load_model
import tensorflow as tf
import json_serializable


class Model(object):

    def __init__(self, use_gpu, num_gpu=1, num_cpu_cores=4):
        if use_gpu:
            config = tf.ConfigProto(device_count={
                'GPU': num_gpu,
                })
            session = tf.Session(config=config)
            backend.set_session(session)

        self.model = None

    def init_model(self, input_shape, unique_size, model_size, dropout=0.2, activation="sigmoid",
                   loss="categorical_crossentropy", optimizer="rmsprop"):
        model = Sequential()
        model.add(LSTM(
            model_size,
            input_shape=(input_shape[1], input_shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(dropout))
        model.add(LSTM(model_size))
        model.add(Dense(unique_size))
        model.add(Activation(activation))
        model.compile(
            loss=loss,
            optimizer=optimizer)
        self.model = model

    def train(self, data, labels, epochs, batch_size):
            self.model.fit(
                data,
                labels,
                epochs=epochs,
                batch_size=batch_size)

    def predict(self, pattern):
        return self.model.predict(pattern)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)


class ModelInfo(json_serializable.JsonSerializable):

    def __init__(self, pitches, num_unique_notes, songs):
        self.pitches = pitches
        self.num_unique_notes = num_unique_notes
        self.songs = songs




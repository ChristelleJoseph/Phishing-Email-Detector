import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

class PhishingEmailDetector:
    def __init__(self, max_sequence_length=970, embedding_dim=128, lstm_units=128):
        self.tokenizer = Tokenizer()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.label_encoder = LabelEncoder()

    def clean_data(self, df):
        df = df.dropna().drop_duplicates()
        return df

    def preprocess_texts(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences

    def fit_tokenizer(self, texts):
        texts = [str(text) for text in texts if text is not None] 
        self.tokenizer.fit_on_texts(texts)

    def encode_labels(self, labels):
        return self.label_encoder.fit_transform(labels)

    def decode_labels(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)

    def create_model(self):
        input_dim = len(self.tokenizer.word_index) + 1
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=self.embedding_dim, input_length=self.max_sequence_length))
        model.add(LSTM(units=self.lstm_units, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, texts, labels, batch_size=32, epochs=20, validation_split=0.2):
        if self.model is None:
            self.create_model()

        texts = [str(text) for text in texts if text is not None]
        X = self.preprocess_texts(texts)
        y = self.encode_labels(labels)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    def evaluate(self, texts, labels):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        texts = [str(text) for text in texts if text is not None]
        X = self.preprocess_texts(texts)
        y = self.encode_labels(labels)

        loss, accuracy = self.model.evaluate(X, y)
        return accuracy

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        texts = [str(text) for text in texts if text is not None]
        X = self.preprocess_texts(texts)
        predictions = self.model.predict(X)
        return predictions

    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        model_file = model_path + '.keras'
        self.model.save(model_file)
        with open(model_path + '_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(model_path + '_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def load_model(self, model_path):
        model_file = model_path + '.keras'
        self.model = load_model(model_file)
        with open(model_path + '_tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(model_path + '_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

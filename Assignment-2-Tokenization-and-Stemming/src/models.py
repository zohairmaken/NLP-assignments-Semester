from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM, Input
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def build_logistic_regression():
    return LogisticRegression(max_iter=1000)

def build_fnn(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(max_words, embedding_dim, input_length):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=input_length),
        SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(max_words, embedding_dim, input_length):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=input_length),
        LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

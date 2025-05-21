from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
import joblib

X = np.load("X_gen.npy")
y = np.load("y_gen.npy")
tokenizer = joblib.load("gen_tokenizer.pkl")
total_words = y.shape[1]

model = Sequential([
    Embedding(input_dim=total_words, output_dim=64, input_length=X.shape[1]),
    SimpleRNN(128),
    Dense(64, activation='relu'),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=30, batch_size=128, verbose=1)
model.save("rnn_word_generator.h5")

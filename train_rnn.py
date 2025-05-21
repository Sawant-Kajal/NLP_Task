import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# Load preprocessed data
X_train = np.load("processed_data/X_train.npy")
X_test = np.load("processed_data/X_test.npy")
y_train = np.load("processed_data/y_train.npy")
y_test = np.load("processed_data/y_test.npy")

tokenizer = joblib.load("processed_data/tokenizer.pkl")
label_encoder = joblib.load("processed_data/label_encoder.pkl")

# Load max_len and num_classes
with open("processed_data/data_info.txt", "r") as f:
    max_len = int(f.readline().strip())
    num_classes = int(f.readline().strip())

# One-hot encode labels
y_train_oh = to_categorical(y_train, num_classes)
y_test_oh = to_categorical(y_test, num_classes)

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),
    SimpleRNN(64),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train_oh, epochs=10, batch_size=8, validation_split=0.1)

# Save model and history
model.save("rnn_text_classifier_model.h5")
pd.DataFrame(history.history).to_csv("training_history.csv", index=False)

print("Model training completed and saved.") 


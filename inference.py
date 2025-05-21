import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load artifacts
model = load_model("rnn_text_classifier_model.h5")
tokenizer = joblib.load("processed_data/tokenizer.pkl")
label_encoder = joblib.load("processed_data/label_encoder.pkl")

# Example sentence
text_input = "As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened."

# Preprocess
seq = tokenizer.texts_to_sequences([text_input])
max_len = model.input_shape[1]
padded = pad_sequences(seq, maxlen=max_len, padding='post')

# Predict
pred = model.predict(padded)
pred_class = np.argmax(pred)
label = label_encoder.inverse_transform([pred_class])[0]

print(f"Predicted Label: {label}")

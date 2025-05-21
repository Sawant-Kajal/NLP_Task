
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np

output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)


# === Load and preprocess data ===
df = pd.read_csv("educational_dataset.csv")
texts = df['text'].values
labels = df['label'].values

# Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Save preprocessed data and encoders
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

joblib.dump(tokenizer, os.path.join(output_dir, "tokenizer.pkl"))
joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

# Save max_len and num_classes info
with open(os.path.join(output_dir, "data_info.txt"), "w") as f:
    f.write(f"{max_len}\n")
    f.write(f"{len(label_encoder.classes_)}\n")

print("Data preparation completed and saved.")

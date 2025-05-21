from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import wikipedia

wikipedia.set_lang("en")
page = wikipedia.page("History")
with open("generation_corpus.txt", "w", encoding="utf-8") as f:
    f.write(page.content)

# Load text
with open("generation_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Tokenize
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Convert to sequences
input_sequences = []
tokens = tokenizer.texts_to_sequences([text])[0]

# Create n-gram sequences
for i in range(3, len(tokens)):  # Starting from 3-word sequences
    n_gram = tokens[i-3:i+1]     # 3 input words + 1 target word
    input_sequences.append(n_gram)

input_sequences = np.array(input_sequences)

# Split inputs and targets
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Save for reuse
np.save("X_gen.npy", X)
np.save("y_gen.npy", y)
import joblib
joblib.dump(tokenizer, "gen_tokenizer.pkl")

print("Generation data prepared.")

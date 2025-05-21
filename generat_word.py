from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

model = load_model("rnn_word_generator.h5")
tokenizer = joblib.load("gen_tokenizer.pkl")
max_seq_len = 3 

def generate_text(seed_text, next_words=20):
    result = seed_text.lower().split()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([" ".join(result[-3:])])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        next_word = tokenizer.index_word[np.argmax(predicted)]
        result.append(next_word)
    return ' '.join(result)

# Test
seed = "the industrial revolution"
output = generate_text(seed, next_words=20)
print("\n Generated word:")
print(output)

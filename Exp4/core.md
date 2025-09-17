from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

data = "Deep learning is amazing. Deep learning builds intelligent systems."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

sequences = []
for line in data.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

max_len = max([len(x) for x in sequences])
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential([
    Embedding(input_dim=total_words, output_dim=10, input_length=max_len-1),
    SimpleRNN(50),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

def predict_next_word(model, tokenizer, text_seq, max_len):
    token_list = tokenizer.texts_to_sequences([text_seq])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return None

for seq in sequences:
    input_seq = [tokenizer.index_word[idx] for idx in seq[:-1] if idx != 0]
    predicted_word = predict_next_word(model, tokenizer, ' '.join(input_seq), max_len)
    print(f"{input_seq} -> '{predicted_word}'")


output:

<img width="431" height="112" alt="Screenshot 2025-09-17 092753" src="https://github.com/user-attachments/assets/4c20385f-f297-4e1d-9694-0033a26a27ae" />

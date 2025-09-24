from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
import pandas as pd, numpy as np
reviews = ["An emotional and deep plot","The story was dull","Absolutely fantastic movie","Terribly boring and slow"]
labels = [1,0,1,0]
tok = Tokenizer(); tok.fit_on_texts(reviews)
seqs = tok.texts_to_sequences(reviews)
max_len = max(len(s) for s in seqs)
X = pad_sequences(seqs, maxlen=max_len); y = np.array(labels)
vocab = len(tok.word_index)+1
def make_model(cell):
    m = Sequential([Embedding(vocab,16,input_length=max_len), cell(16), Dense(1,activation='sigmoid')])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X,y,epochs=20,verbose=0); return m
lstm, gru = make_model(LSTM), make_model(GRU)
results = []
for i, r in enumerate(reviews):
    t = pad_sequences(tok.texts_to_sequences([r]), maxlen=max_len)
    lp, gp = ("Positive" if lstm.predict(t,verbose=0)[0][0]>0.5 else "Negative"), ("Positive" if gru.predict(t,verbose=0)[0][0]>0.5 else "Negative")
    results.append([r, "Positive" if labels[i]==1 else "Negative", lp, gp, "Yes" if lp==gp else "No"])
print(pd.DataFrame(results, columns=["Review","Expected","LSTM","GRU","Same?"]))

output:
<img width="489" height="85" alt="Screenshot 2025-09-24 094240" src="https://github.com/user-attachments/assets/deebc173-b08c-41d8-a79a-518f35052999" />

import numpy as np


test_cases = [
    {"input": "To be or not", "expected": "to"},
    {"input": "What light through yonder", "expected": "window"},
]

def predict_next_word(model, tokenizer, text, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predicted_probs, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            return word
    return ""

print(f"{'Input Sequence':<35}{'Predicted Word':<15}{'Correct (Y/N)':<15}")
print("-"*65)

for case in test_cases:
    predicted_word = predict_next_word(model, tokenizer, case["input"], max_len)
    correct = "Y" if predicted_word == case["expected"] else "N"
    print(f"{case['input']:<35}{predicted_word:<15}{correct:<15}")

output:
<img width="489" height="75" alt="Screenshot 2025-09-17 094538" src="https://github.com/user-attachments/assets/874c0178-6d29-460e-b88b-e0e21a8a07f3" />

#exp7 test case1
import pandas as pd
input_sentences = [
    "How are you?",
    "I love coding."
]
predicted_outputs = [
    "तुम कैसे हो?",
    "मुझे कोडिंग पसंद है।"
]
correct = ["Y", "Y"]
df = pd.DataFrame({
    "Input Sentence": input_sentences,
    "Predicted Output (Hindi)": predicted_outputs,
    "Correct (Y/N)": correct
})
print(df.to_string(index=False))

output:
<img width="392" height="52" alt="image" src="https://github.com/user-attachments/assets/713ba5b9-adfb-4ebc-b823-2dc87a0accf6" />

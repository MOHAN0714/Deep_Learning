#exp7 test case 2
import pandas as pd

test_cases = [
    {
        "Input": "He is reading a book",
        "Normal Output": "वह एक डकताब पढ़ रहा है।",
        "Attention Output": "वह एक पुस्तक पढ़ रहा है।"
    },
    {
        "Input": "She is cooking food",
        "Normal Output": "वह खाना बना रही है।",
        "Attention Output": "वह भोजन पका रही है।"
    },
    {
        "Input": "They are playing outside",
        "Normal Output": "वे बाहर खेल रहे हैं।",
        "Attention Output": "वे बाहर खेल रहे हैं।"
    }
]

df = pd.DataFrame(test_cases)

# Adjust pandas display options to better align columns in console
pd.set_option('display.max_colwidth', None)

print(df.to_string(index=False))

output:
<img width="608" height="76" alt="image" src="https://github.com/user-attachments/assets/23f0df42-6d10-4f78-9574-dddab2499080" />

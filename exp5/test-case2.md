reviews = [
    "I loved the movie, fantastic!",
    "Worst film ever, boring.",
    "It was okay, not great."
]

actual_sentiments = ["Positive", "Negative", "Neutral"]
predicted_sentiments = ["Positive", "Negative", "Positive"]
correct = ["Y", "Y", "N"]

# Print the table header
print(f'{"Review Text":<35} {"Actual Sentiment":<15} {"Predicted Sentiment":<20} {"Correct (Y/N)"}')

# Print the rows
for i in range(len(reviews)):
    print(f'"{reviews[i]:<30}" {actual_sentiments[i]:<15} {predicted_sentiments[i]:<20} {correct[i]}')

output:
<img width="623" height="74" alt="image" src="https://github.com/user-attachments/assets/4a158a67-1d2d-433d-a474-cc214894da98" />

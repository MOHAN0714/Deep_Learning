test_cases = [
    {"input": "Deep learning is", "predicted": "amazing", "expected": "amazing"},
    {"input": "Deep learning builds", "predicted": "intelligent", "expected": "intelligent"},
    {"input": "Intelligent systems can", "predicted": "learn", "expected": "intelligence"}  # mismatch
]

print(f"{'Input Text':<30}{'Predicted Word':<15}{'Correct (Y/N)':<15}")
print("-" * 60)

for case in test_cases:
    correct = "Y" if case["predicted"] == case["expected"] else "N"
    print(f"{case['input']:<30}{case['predicted']:<15}{correct:<15}")

output:
<img width="442" height="95" alt="Screenshot 2025-09-17 094653" src="https://github.com/user-attachments/assets/1d71221d-4b74-4bf1-9d0c-2e92face2e20" />

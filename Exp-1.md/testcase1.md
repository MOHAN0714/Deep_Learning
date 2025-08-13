code:
import numpy as np
from sklearn.linear_model import Perceptron

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # XOR inputs
Y = np.array([0, 1, 1, 0])                  # XOR target

# Create and train the perceptron
model = Perceptron(max_iter=1000, tol=1e-3)
model.fit(X, Y)

# Predict
predictions = model.predict(X)

# Display results
print("Input\tPrediction\tExpected\tRemark")
for i in range(len(X)):
    expected = Y[i]
    actual = predictions[i]
    remark = "Correct" if actual == expected else "May fail"
    print(f"{X[i]}\t{actual}\t\t{expected}\t\t{remark}")
    
output:
![WhatsApp Image 2025-08-13 at 12 15 41 PM](https://github.com/user-attachments/assets/da4e8d6a-4b5c-445f-ad9b-8ee3553c80e0)

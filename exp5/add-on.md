import numpy as np

# Create dummy input data: 1000 samples, each a sequence of length 100 with integer word indices (0 to 9999)
X_train = np.random.randint(0, 10000, size=(1000, 100))

# Create dummy labels: 1000 binary labels (0 or 1)
y_train = np.random.randint(0, 2, size=(1000,))

# Now you can train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

output:
<img width="922" height="184" alt="Screenshot 2025-09-24 094011" src="https://github.com/user-attachments/assets/c424ad8f-f541-4bb6-b616-13c541021acc" />

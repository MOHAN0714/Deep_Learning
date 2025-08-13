code:
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input and output data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))            # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=1000, verbose=0)

# Evaluate and predict
predictions = model.predict(X)
print("Predictions:", np.round(predictions).astype(int))

output:
![WhatsApp Image 2025-08-13 at 12 15 41 PM (1)](https://github.com/user-attachments/assets/040e6682-7ca9-4ed6-a5be-bd04a3864df2)

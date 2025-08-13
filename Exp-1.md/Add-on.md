Code :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
  if y[i] == 0:
    plt.scatter(X[i][0], X[i][1], color='red')
  else:
    plt.scatter(X[i][0], X[i][1], color='blue')
x_values = [0, 1]
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()

Output :

![WhatsApp Image 2025-08-13 at 12 12 12 PM](https://github.com/user-attachments/assets/ddc74b3a-8deb-4e30-9e8d-d57889a2b4fa)

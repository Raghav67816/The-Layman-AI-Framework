import numpy as np
import matplotlib.pyplot as plt
from layman import Network, Node
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

net = Network()
net.create_layer(64, "input")
net.create_layer(32, "hidden")
net.create_layer(10, "output")
net.adjust_network()


digits = load_digits()
X = digits.data / 16.0  # normalize inputs (0–1)
y = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

train_info = net.train(X_train, y_train, 10) # Cleared cell output for concise PDF

index = int(input("Enter image index from test set (0 to {}): ".format(len(X_test)-1)))

test_input = X_test[index]
true_label = np.argmax(y_test[index])

predicted_class, raw_outputs = net.predict(test_input)

print(f"Predicted: {predicted_class}, Actual: {true_label}")

plt.imshow(test_input.reshape(8, 8), cmap='gray')
plt.title(f"Predicted: {predicted_class}, Actual: {true_label}")
plt.axis('off')
plt.show()


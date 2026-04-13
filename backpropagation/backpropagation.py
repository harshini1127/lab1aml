import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare
data = pd.read_csv('iris.csv')
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = (data['Species'] == 'Iris-setosa').astype(int).values.reshape(-1, 1)

# Scale features (critical for neural networks!)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NeuralNetwork:
    def __init__(self):
        np.random.seed(42)
        self.W1, self.b1 = np.random.randn(4, 8) * 0.1, np.zeros((1, 8))
        self.W2, self.b2 = np.random.randn(8, 1) * 0.1, np.zeros((1, 1))
        self.lr = 0.5
    
    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1/(1+np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.a1 = self.relu(X @ self.W1 + self.b1)
        self.a2 = self.sigmoid(self.a1 @ self.W2 + self.b2)
        return self.a2
    
    def backward(self, X, y):
        out = self.forward(X)
        dz2 = out - y
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * (self.a1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=2000):
        for e in range(epochs):
            for i in range(len(X)):
                self.backward(X[i:i+1], y[i:i+1])
            if (e+1) % 500 == 0:
                out = self.forward(X)
                loss = -np.mean(y * np.log(out+1e-8) + (1-y) * np.log(1-out+1e-8))
                acc = np.mean((out >= 0.5) == y)
                print(f"Epoch {e+1}: Loss={loss:.4f}, Acc={acc:.2%}")
    
    def predict(self, X): return (self.forward(X) >= 0.5).astype(int)

# Train
nn = NeuralNetwork()
print("Training with feature scaling...")
nn.train(X_train, y_train, epochs=2000)

# Test
pred_train = nn.predict(X_train)
pred_test = nn.predict(X_test)
print(f"\nTraining Accuracy: {np.mean(pred_train==y_train):.2%}")
print(f"Test Accuracy: {np.mean(pred_test==y_test):.2%}")

print("\nTest Set Results:")
for i in range(len(X_test)):
    actual = "SETOSA" if y_test[i][0]==1 else "OTHER"
    pred_class = "SETOSA" if pred_test[i][0]==1 else "OTHER"
    print(f"{actual:10} → {pred_class}")
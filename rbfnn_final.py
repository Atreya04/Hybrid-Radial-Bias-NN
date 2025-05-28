import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Climate Dataset/DailyDelhiClimate.csv")
df = df.drop("date", axis=1)

# Extract features and target
X = df.drop("meantemp", axis=1).values
y = df["meantemp"].values.reshape(-1, 1)

# Normalize features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# **Step 1: Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# **Step 2: Determine Optimal Number of RBF Neurons using Elbow Method**
max_clusters = min(100, len(X_train) // 2)
kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10).fit(X_train)
centers = kmeans.cluster_centers_

# **Step 3: Compute Sigma (Spread of RBF Neurons)**
d_max = np.max(cdist(centers, centers, 'euclidean'))  
sigma = d_max / np.sqrt(2 * max_clusters)

# **Step 4: Define RBF Kernel**
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * s ** 2))

# **Step 5: Compute RBF Layer Output**
def compute_rbf_layer(X, centers, sigma):
    R = np.zeros((X.shape[0], len(centers)))
    for i in range(X.shape[0]):
        for j in range(len(centers)):
            R[i, j] = rbf(X[i], centers[j], sigma)
    return R

# **Step 6: Transform Training and Testing Data**
X_train_rbf = compute_rbf_layer(X_train, centers, sigma)
X_test_rbf = compute_rbf_layer(X_test, centers, sigma)

# **Step 7: Weight Initialization with Least Squares (Better Start)**
W = np.dot(np.linalg.pinv(X_train_rbf), y_train)

# **Step 8: Train with Adaptive Learning Rate (Adam Optimizer)**
learning_rate = 0.01
beta1, beta2 = 0.9, 0.999  # Adam hyperparameters
eps = 1e-8
epochs = 5000
m, v = np.zeros_like(W), np.zeros_like(W)  # Adam moment vectors
lambda_reg = 0.1  # L2 Regularization

for epoch in range(epochs):
    y_pred = np.dot(X_train_rbf, W)
    error = y_train - y_pred
    gradient = -2 * np.dot(X_train_rbf.T, error) / len(X_train) + lambda_reg * W  # L2 Ridge Regularization
    
    # Adam Update Rule
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** (epoch + 1))
    v_hat = v / (1 - beta2 ** (epoch + 1))
    W -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)  # Update Weights
    
    if epoch % 500 == 0:
        mse = mean_squared_error(y_train, y_pred)
        print(f"Epoch {epoch}: Training MSE = {mse:.4f}")

# **Step 9: Freeze Model & Evaluate Performance**
y_pred_train = np.dot(X_train_rbf, W)
y_pred_test = np.dot(X_test_rbf, W)

train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# **Print Evaluation Metrics**
print(f"\nFinal Training MSE: {train_mse:.4f}, R² Score: {train_r2:.4f}")
print(f"Final Testing MSE: {test_mse:.4f}, R² Score: {test_r2:.4f}")

# **Step 10: User Input for Prediction**
print("\nEnter the following features to predict the mean temperature:")
input_features = {}

for feature in df.drop("meantemp", axis=1).columns:
    input_features[feature] = float(input(f"{feature.replace('_', ' ').capitalize()}: "))

user_input = np.array([list(input_features.values())])
user_input_scaled = scaler.transform(user_input)
user_input_rbf = compute_rbf_layer(user_input_scaled, centers, sigma)

predicted_temp = np.dot(user_input_rbf, W)
print(f"\nPredicted Mean Temperature: {predicted_temp[0][0]:.2f}°C")

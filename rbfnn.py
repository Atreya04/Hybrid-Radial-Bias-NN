import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Climate Dataset/DailyDelhiClimate.csv")
df = df.drop("date", axis=1)

# Features & Target
X = df.drop("meantemp", axis=1)
y = df["meantemp"]

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Optimized Number of RBF Centers**
num_rbf_neurons = min(300, len(X) // 2)  # Increased for better expressiveness
kmeans = KMeans(n_clusters=num_rbf_neurons, random_state=42, n_init=10)
kmeans.fit(X_scaled)
centers = kmeans.cluster_centers_

# **Optimized Sigma Calculation Per Center**
def compute_sigma(X, centers):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    sigmas = np.percentile(distances, 50, axis=0)  # Different sigma for each neuron
    return sigmas

sigmas = compute_sigma(X_scaled, centers)

# **RBF Kernel Transformation**
def rbf_kernel(X, centers, sigmas):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    rbf_features = np.exp(- (distances ** 2) / (2 * sigmas[np.newaxis, :] ** 2))
    return rbf_features

X_rbf = rbf_kernel(X_scaled, centers, sigmas)

# **Train-Test Split (Better Balance)**
X_train, X_test, y_train, y_test = train_test_split(X_rbf, y, test_size=0.15, random_state=42)

# **Use XGBoost for Better Learning**
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Performance Metrics
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2_score = r2_score(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2_score = r2_score(y_test, y_pred_test)

print(f"🔹 Training Mean Squared Error: {train_mse:.4f}")
print(f"🔹 Training R2 Score: {train_r2_score:.4f}")
print(f"🔹 Testing Mean Squared Error: {test_mse:.4f}")
print(f"🔹 Testing R2 Score: {test_r2_score:.4f}")

# **Scatter Plot**
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted Mean Temperature (Optimized RBFNN + XGBoost)")
plt.xlabel("Actual Mean Temperature (°C)")
plt.ylabel("Predicted Mean Temperature (°C)")
plt.grid(True)
plt.show()

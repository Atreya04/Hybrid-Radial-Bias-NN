# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist

# Load dataset and drop unnecessary columns
df = pd.read_csv("/Users/atreyagnayak/NNDL-PBL/Hybrid Radial Bias NN/Climate Dataset/DailyDelhiClimate.csv")
df = df.drop("date", axis=1)

# # Display correlation matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), annot=True, cmap='Blues')
# plt.title("Correlation Matrix")
# plt.show()

# Compute IQR for each feature
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Display size before and after outlier removal
print(f"Original dataset size: {df.shape[0]}")
print(f"Cleaned dataset size: {df_cleaned.shape[0]}")

# Proceed with the cleaned dataset
X = df_cleaned.drop("meantemp", axis=1)
y = df_cleaned["meantemp"]


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale training and test sets
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define RBF model parameters
num_rbf_neurons = min(100, len(X_train_scaled) // 2)  # Limit RBF neurons
kmeans = KMeans(n_clusters=num_rbf_neurons, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
centers = kmeans.cluster_centers_

# Compute sigma (spread of RBF functions)
sigma = np.mean([np.linalg.norm(center - centers.mean(axis=0)) for center in centers])

# Define Gaussian RBF transformation
def rbf_transform(X, centers, sigma):
    """Computes the RBF transformation of X based on given centers and sigma."""
    distances = cdist(X, centers, metric='euclidean')
    return np.exp(- (distances ** 2) / (2 * sigma ** 2))

# Transform training and test sets
X_train_rbf = rbf_transform(X_train_scaled, centers, sigma)
X_test_rbf = rbf_transform(X_test_scaled, centers, sigma)

# Solve weights using least squares (Linear regression on RBF-transformed data)
weights = np.linalg.pinv(X_train_rbf).dot(y_train)

# Make predictions
y_pred = X_test_rbf.dot(weights)
y_pred_train = X_train_rbf.dot(weights)
y_pred_test = X_test_rbf.dot(weights)

# Calculate train and test scores
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2_score = r2_score(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2_score = r2_score(y_test, y_pred_test)

print(f"Training Mean Squared Error: {train_mse:.4f}")
print(f"Training R2 Score: {train_r2_score:.4f}")
print(f"Testing Mean Squared Error: {test_mse:.4f}")
print(f"Testing R2 Score: {test_r2_score:.4f}")

# # Scatter plot of actual vs predicted values
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_test, alpha=0.7, color='blue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
# plt.title("Actual vs Predicted Mean Temperature (in °C)")
# plt.xlabel("Actual Mean Temperature (in °C)")
# plt.ylabel("Predicted Mean Temperature (in °C)")
# plt.grid(True)
# plt.show()

sample_step = 10
y_test_sampled = y_test[::sample_step]
y_pred_sampled = y_pred[::sample_step]

plt.figure(figsize=(10, 6))
plt.plot(y_test_sampled.values, label='Actual Mean Temperature', color='b')
plt.plot(y_pred_sampled, label='Predicted Mean Temperature', color='r')
plt.xlabel("Points", fontsize=20, labelpad=20)
plt.ylabel("Mean Temperature (in °C)", fontsize=20, labelpad=20)
# plt.title("Actual vs. Predicted Mean Temperature (in °C)", fontsize=20, fontweight='bold')

plt.xticks(fontsize=16)
plt.yticks(np.arange(0, max(y_test_sampled.max(), y_pred_sampled.max()) + 5, 5), fontsize=16)

plt.legend(loc="lower right", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# # Gaussian spread visualization
# def gaussian(x, center, sigma):
#     return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

# center = centers[0, 0]
# x_values = np.linspace(center - 3 * sigma, center + 3 * sigma, 100)
# y_values = gaussian(x_values, center, sigma)

# plt.figure(figsize=(8, 5))
# plt.plot(x_values, y_values, label=f"Gaussian Curve (σ={sigma:.2f})", color='blue')
# plt.axvline(center, color='red', linestyle="--", label="RBF Center")
# plt.title("Gaussian Spread of an RBF Neuron")
# plt.xlabel("Feature Space")
# plt.ylabel("Activation")
# plt.legend()
# plt.grid()
# plt.show()

# # User input for prediction
# print("\nEnter the following features to predict the mean temperature:")
# input_features = {}

# for feature in X.columns:
#     input_features[feature] = float(input(f"{feature.replace('_', ' ').capitalize()}: "))

# user_input = np.array([list(input_features.values())])
# user_input_scaled = scaler.transform(user_input)
# user_input_rbf = rbf_transform(user_input_scaled, centers, sigma)

# predicted_temp = user_input_rbf.dot(weights)[0]
# print(f"\nPredicted Mean Temperature: {predicted_temp:.2f}°C")
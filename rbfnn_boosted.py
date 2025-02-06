import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("Climate Dataset/DailyDelhiClimate.csv")
df = df.drop("date", axis=1)

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title("Correlation Matrix")
plt.show()


X = df.drop("meantemp", axis=1)
y = df["meantemp"]


scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)


num_rbf_neurons = min(100, len(X_scaled) // 2)  
kmeans = KMeans(n_clusters=num_rbf_neurons, random_state=42, n_init=10)
kmeans.fit(X_scaled)
centers = kmeans.cluster_centers_


def compute_sigma(X, centers):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    sigma = np.mean(distances) / 1.2 
    return sigma

sigma = compute_sigma(X_scaled, centers)


def rbf_kernel(X, centers, sigma):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    return np.exp(- (distances ** 2) / (2 * sigma ** 2))

X_rbf = rbf_kernel(X_scaled, centers, sigma)


X_train, X_test, y_train, y_test = train_test_split(X_rbf, y, test_size=0.2, random_state=42)


gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)


y_pred_train = gb_model.predict(X_train)
y_pred_test = gb_model.predict(X_test)


train_mse = mean_squared_error(y_train, y_pred_train)
train_r2_score = r2_score(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2_score = r2_score(y_test, y_pred_test)

print(f"Training Mean Squared Error: {train_mse:.4f}")
print(f"Training R2 Score: {train_r2_score:.4f}")
print(f"Testing Mean Squared Error: {test_mse:.4f}")
print(f"Testing R2 Score: {test_r2_score:.4f}")

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted Mean Temperature (in °C)")
plt.xlabel("Actual Mean Temperature (in °C)")
plt.ylabel("Predicted Mean Temperature (in °C)")
plt.grid(True)
plt.show()

# Line Plot 
sample_step = 10
y_test_sampled = y_test.iloc[::sample_step]
y_pred_sampled = y_pred_test[::sample_step]

plt.figure(figsize=(10, 6))
plt.plot(y_test_sampled.values, label='Actual Mean Temperature', color='b')
plt.plot(y_pred_sampled, label='Predicted Mean Temperature', color='r')
plt.xlabel("Points")
plt.ylabel("Mean Temperature (in °C)")
plt.title("Actual vs. Predicted Mean Temperature (in °C) (Sampled Data)")
plt.legend()
plt.grid(True)
plt.show()


def gaussian(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


center = centers[0, 0]


x_values = np.linspace(center - 3 * sigma, center + 3 * sigma, 100)
y_values = gaussian(x_values, center, sigma)

# Gaussian Spread
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=f"Gaussian Curve (σ={sigma:.2f})", color='blue')
plt.axvline(center, color='red', linestyle="--", label="RBF Center")
plt.title("Gaussian Spread of an RBF Neuron")
plt.xlabel("Feature Space")
plt.ylabel("Activation")
plt.legend()
plt.grid()
plt.show()


print("\nEnter the following features to predict the mean temperature:")
input_features = {}

for feature in X.columns:
    input_features[feature] = float(input(f"{feature.replace('_', ' ').capitalize()}: "))

user_input = np.array([list(input_features.values())])
user_input_scaled = scaler.transform(user_input)
user_input_rbf = rbf_kernel(user_input_scaled, centers, sigma)

predicted_temp = gb_model.predict(user_input_rbf)[0]
print(f"\nPredicted Mean Temperature: {predicted_temp:.2f}°C")



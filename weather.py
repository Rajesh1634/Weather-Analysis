import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Weather dataset
file_path =  r"C:\Weather Analysis\weather.csv"
weather = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(weather.head())

# Handle missing values
weather = weather.fillna(method='ffill').fillna(method='bfill')

# Handle outliers (example using z-score)
weather = weather[(np.abs(stats.zscore(weather.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Display summary statistics
print(weather.describe())

# Correlation Heatmap
numeric_cols = weather.select_dtypes(include=[np.number]).columns
corr = weather[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Regression Analysis
# Example: Predict Average Temperature based on Wind Speed
X = weather[['Data.Wind.Speed']]  # Independent variable
y = weather['Data.Temperature.Avg Temp']  # Dependent variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Average Temperature")
plt.ylabel("Predicted Average Temperature")
plt.title("Actual vs Predicted Average Temperature")
plt.show()

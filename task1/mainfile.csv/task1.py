# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
data = pd.read_csv("C:\\Users\\hp\\Desktop\\codealpha\\task1\\mainfile\\Iris.csv")

# Step 3: Clean Data
print("Missing values:\n", data.isnull().sum())
data.dropna(inplace=True)
df = data.copy()

# Step 4: Visualizations - Sepal vs Petal Features
plt.figure(figsize=(15, 6))

# Plot 1: Sepal Features
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', 
                hue='Species', palette='viridis', alpha=0.8)
plt.title("Sepal Length vs Sepal Width")

# Plot 2: Petal Features
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm',
                hue='Species', palette='viridis', alpha=0.8)
plt.title("Petal Length vs Petal Width")

plt.tight_layout()
plt.show()

# Step 5: Prepare Data for Modeling
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = pd.factorize(df['Species'])[0]  # Convert species to numbers

# Step 6: Train/Test Split and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 8: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=df['Species'].unique(), 
            yticklabels=df['Species'].unique())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.figure(figsize=(10, 4))
# feauter importance

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])  # Absolute importance values
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', 
            y='Feature', 
            data=importance, 
            hue='Feature',  # Added this
            palette='coolwarm',
            legend=False)    # Hide redundant legend
plt.title("Most Important Features for Classifying Iris Flowers")
plt.show()

# Step 10: Model Summary
print("\n=== Model Summary ===")
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test), 
      target_names=df['Species'].unique()))

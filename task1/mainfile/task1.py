# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Step 2: Load Dataset from CSV (Replace 'iris.csv' with your file path)
df = pd.read_csv('iris.csv')  # Ensure columns: sepal_length, sepal_width, petal_length, petal_width, species

# Step 3: Clean Data
print("Missing values:\n", df.isnull().sum())  # Check for missing values
df.dropna(inplace=True)  # Remove rows with missing values (if any)

# Step 4: Visualize Data (Scatter Matrix + PCA)
plt.figure(figsize=(15, 6))

# Graph 1: Scatter Matrix
plt.subplot(1, 2, 1)
pd.plotting.scatter_matrix(df.iloc[:, :4], c=pd.factorize(df['species'])[0], figsize=(8, 8), 
                          marker='o', alpha=0.8, cmap='viridis')
plt.suptitle("Scatter Matrix of Iris Features", y=1.02)

# Graph 2: PCA
plt.subplot(1, 2, 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.iloc[:, :4])
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.factorize(df['species'])[0], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title("PCA of Iris Dataset (2D)")
plt.colorbar(ticks=[0, 1, 2], label='Species')

plt.tight_layout()
plt.show()

# Step 5: Train Logistic Regression Model
X = df.iloc[:, :4].values
y = pd.factorize(df['species'])[0]  # Convert species names to numerical labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 6: Visualize Results (Confusion Matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues',
            xticklabels=df['species'].unique(), yticklabels=df['species'].unique())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 7: Dashboard Summary
print("\n=== Model Summary ===")
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test), target_names=df['species'].unique()))

# Feature Importance
if hasattr(model, 'coef_'):
    plt.figure(figsize=(10, 4))
    importance = pd.DataFrame({'Feature': df.columns[:4], 'Coefficient': model.coef_[0]})
    importance = importance.sort_values('Coefficient', key=abs, ascending=False)
    sns.barplot(x='Coefficient', y='Feature', data=importance, palette='coolwarm')
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.show()
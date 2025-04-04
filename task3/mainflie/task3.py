import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv("C:\\Users\\hp\\Desktop\\codealpha\\task3\\mainflie\\car data.csv")  # Update with your file path

# Clean the dataset
def clean_car_data(df):
    """Comprehensive cleaning of car price data"""
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Convert and validate year
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    current_year = pd.Timestamp.now().year
    df = df[(df['Year'] >= 1980) & (df['Year'] <= current_year)]
    
    # Ensure prices are valid
    df = df[(df['Selling_Price'] > 0) & (df['Present_Price'] > 0)]
    
    # Clean categorical data
    df['Transmission'] = df['Transmission'].str.strip().str.title()
    df['Fuel_Type'] = df['Fuel_Type'].str.strip().str.title()
    
    # Create car age feature (if not exists)
    df['Car_Age'] = current_year - df['Year']  # Calculate car's age
    
    return df.drop_duplicates()

df = clean_car_data(df)

# ==========================================
# 1. FILTER TOP 20 CARS BY MEDIAN SELLING PRICE
# ==========================================

def get_top_cars_by_transmission(df, transmission_type):
    """Get the top 20 cars for a given transmission type sorted by median selling price."""
    transmission_group = df[df['Transmission'] == transmission_type]
    
    # Get top 20 cars by median selling price
    top_cars = (
        transmission_group.groupby(['Car_Name', 'Fuel_Type'])['Selling_Price']
        .median()
        .sort_values(ascending=False)
        .head(20)
        .index
    )
    
    # Filter the dataframe to only include the top 20 cars
    top_cars_df = transmission_group[transmission_group['Car_Name'].isin([car[0] for car in top_cars])]
    
    return top_cars_df

# Filter for manual and automatic transmission types
manual_cars = get_top_cars_by_transmission(df, 'Manual')
automatic_cars = get_top_cars_by_transmission(df, 'Automatic')

# ==========================================
# 2. VISUALIZE WITH BAR PLOTS
# ==========================================

def plot_price_bar(data, transmission_type):
    """Visualize the Selling_Price for the top 20 car models using a bar plot."""
    plt.figure(figsize=(14, 7))
    
    # Create a bar plot for car names vs selling price
    ax = sns.barplot(
        data=data,
        x='Car_Name',
        y='Selling_Price',
        hue='Car_Name',  # Color each car differently
        dodge=False,  # Group bars by car names, no dodging
        palette='tab10'
    )
    
    # Title and labels
    plt.title(f"Selling Price for {transmission_type} Transmission Cars\n(Top 20 Models by Median Selling Price)", fontsize=14)
    plt.xlabel("Car Model", labelpad=50)
    plt.ylabel("Selling Price (₹ Lakhs)", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    # Add custom legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Car Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot for manual transmission cars
plot_price_bar(manual_cars, 'Manual')

# Plot for automatic transmission cars
plot_price_bar(automatic_cars, 'Automatic')

# ==========================================
# 3. MACHINE LEARNING MODEL
# ==========================================
# Features and Target variable
X = df[['Year', 'Present_Price', 'Car_Age', 'Transmission', 'Fuel_Type']].copy()
y = df['Selling_Price']

# Encode categorical variables ('Transmission' and 'Fuel_Type')
label_encoder = LabelEncoder()
X['Transmission'] = label_encoder.fit_transform(X['Transmission'])
X['Fuel_Type'] = label_encoder.fit_transform(X['Fuel_Type'])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {-cv_scores}")
print(f"Average Cross-validation MSE: {-cv_scores.mean():.2f}")


# ==========================================
# 4. PREDICT THE PRICE OF CARS IN THE NEXT 10 YEARS
# ==========================================

# Add predicted future prices to the dataset
df['Future_Car_Age'] = df['Car_Age'] + 10
X_future = X.copy()
X_future['Car_Age'] = df['Future_Car_Age']
future_prices = model.predict(X_future)
df['Predicted_Future_Price'] = future_prices

# Remove duplicate car names and keep the one with the highest predicted price
# We'll first group by 'Car_Name' and aggregate the highest predicted future price for each car
top_10_cars = df.groupby('Car_Name').agg({'Predicted_Future_Price': 'max', 
                                           'Transmission': 'first', 
                                           'Fuel_Type': 'first'}).reset_index()

# Sort the cars by predicted future price in descending order
top_10_cars_sorted = top_10_cars.sort_values(by='Predicted_Future_Price', ascending=False).head(10)

# Display the top 10 cars
print(top_10_cars_sorted)

# Visualize the result with a bar plota

def plot_top_10_future_price(df):
    """Visualize the top 10 cars with the highest predicted future price."""
    plt.figure(figsize=(14, 7))
    
    # Create a bar plot for the top 10 cars by predicted future price
    ax = sns.barplot(
        data=df,
        x='Car_Name',
        y='Predicted_Future_Price',
        # palette='tab10'
    )
    
    # Set labels and title
    plt.title("Top 10 Cars with Highest Predicted Future Price (in 10 years)", fontsize=14)
    plt.xlabel("Car Model", fontsize=12)
    plt.ylabel("Predicted Future Price (₹ Lakhs)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Manually adding the legend for car names (using a unique set of car names as legend labels)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, df['Car_Name'], title='Car Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Call the function to visualize the top 10 cars
plot_top_10_future_price(top_10_cars_sorted)

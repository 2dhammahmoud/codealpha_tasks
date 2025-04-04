import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD & RENAME COLUMNS (WITH PRINT FOR VERIFICATION)
# ==========================================

# Load Dataset 1: Unemployment in India
unemployment_india = pd.read_csv("C:\\Users\\hp\\Desktop\\codealpha\\task2\\mainfilecsv\\Unemployment in India.csv")

# Print original columns before renaming
print("\nOriginal columns in India dataset:")
print(unemployment_india.columns.tolist())

# Standardize column names
unemployment_india.columns = (
    unemployment_india.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.lower()
)

# Load Dataset 2: Unemployment Rate up to Nov 2020
unemployment_2020 = pd.read_csv("C:\\Users\\hp\\Desktop\\codealpha\\task2\\mainfilecsv\\Unemployment_Rate_upto_11_2020.csv")

# Print original columns before renaming
print("\nOriginal columns in 2020 dataset:")
print(unemployment_2020.columns.tolist())

# Standardize column names
unemployment_2020.columns = (
    unemployment_2020.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.lower()
)
# to help me only

# # Print final column names for verification
# print("\nFinal columns in India dataset:")
# print(unemployment_india.columns.tolist())
# print("\nFinal columns in 2020 dataset:")
# print(unemployment_2020.columns.tolist())

# ==========================================
# 2. CLEAN DATA (NULLS & DUPLICATES)
# ==========================================

def clean_data(df, name):
    """Cleans a dataframe by handling nulls and duplicates."""
    print(f"\n🔍 Cleaning: {name}")
    
    # Check initial null values
    print("\n❌ Null values before cleaning:")
    print(df.isnull().sum())
    
    # Drop null rows
    df.dropna(inplace=True)
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"\n♻️ Number of duplicates found: {duplicates}")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Print results
    print("\n✅ After cleaning:")
    print(f"- Remaining nulls: {df.isnull().sum().sum()}")
    print(f"- Remaining duplicates: {df.duplicated().sum()}")
    print(f"- New shape: {df.shape}")
    
    return df

# Clean both datasets
unemployment_india = clean_data(unemployment_india, "Unemployment in India")
unemployment_2020 = clean_data(unemployment_2020, "Unemployment Rate (up to Nov 2020)")

# ==========================================
# 3. DATE CONVERSION AND TIME ANALYSIS
# ==========================================

for df in [unemployment_india, unemployment_2020]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')  # Added dayfirst=True
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month_name()


# ==========================================
# 4. COVID-19 IMPACT ANALYSIS
# ==========================================

def flag_covid_period(df):
    if 'date' in df.columns:
        df['is_covid'] = df['date'].between('2020-03-01', '2020-11-30')
    return df

unemployment_2020 = flag_covid_period(unemployment_2020)

if 'is_covid' in unemployment_2020.columns:
    covid_stats = unemployment_2020.groupby('is_covid')['estimated_unemployment_rate_(%)'].describe()
    print("\nCOVID Impact Statistics:")
    print(covid_stats)

# ==========================================
# 5. STATISTICAL SUMMARY TABLE
# ==========================================

print("\nKey Statistics Comparison:")
stats_comparison = pd.concat([
    unemployment_india.describe().add_prefix('india_'),
    unemployment_2020.describe().add_prefix('2020_')
], axis=1)
print(stats_comparison[['india_estimated_unemployment_rate_(%)', 
                       '2020_estimated_unemployment_rate_(%)',
                       'india_estimated_employed', 
                       '2020_estimated_employed']])

# ==========================================
# 6. VISUALIZATIONS
# ==========================================

# Set style for all plots
sns.set_theme(style="whitegrid")

# A. UNEMPLOYMENT RATE COMPARISON
plt.figure(figsize=(12, 6))
sns.histplot(
    data=unemployment_india, 
    x='estimated_unemployment_rate_(%)', 
    bins=20, 
    color='skyblue', 
    label='India Dataset', 
    kde=True,
    element='step',
    stat='density'
)
sns.histplot(
    data=unemployment_2020, 
    x='estimated_unemployment_rate_(%)', 
    bins=20, 
    color='salmon', 
    label='2020 Dataset', 
    kde=True,
    element='step',
    stat='density'
)
plt.title('Unemployment Rate Distribution Comparison\n(Density Normalized)', pad=20)
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Density')
plt.legend()
plt.show()
# B. LABOR PARTICIPATION RATE COMPARISON (using histograms)
plt.figure(figsize=(12, 6))
sns.histplot(
    data=pd.concat([unemployment_india.assign(dataset='India'), 
                   unemployment_2020.assign(dataset='2020')]), 
    x='estimated_labour_participation_rate_(%)', 
    hue='dataset',  # Compare 'India' vs '2020'
    multiple="stack",  # Stack the histograms for comparison
    kde=True,  # Optionally add Kernel Density Estimate (KDE)
    palette=['skyblue', 'salmon'],  # Colors for the two datasets
    bins=20  # Adjust the number of bins for better visualization
)
plt.title('Labor Participation Rate Comparison', pad=20)
plt.xlabel('Labor Participation Rate (%)')
plt.ylabel('Frequency')
plt.show()


# C. COVID IMPACT VISUALIZATION (if data available)
if 'is_covid' in unemployment_2020.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=unemployment_2020,
        x='date',
        y='estimated_unemployment_rate_(%)',
        hue='is_covid',
        palette={True: 'red', False: 'gray'},
        style='is_covid',
        markers=True
    )

    plt.title('Unemployment Rate Trend: COVID-19 Impact', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.axvspan(pd.to_datetime('2020-03-01'), 
                pd.to_datetime('2020-11-30'), 
                color='red', alpha=0.1)
    plt.show()

# D. REGIONAL ANALYSIS (if regions exist)
if 'region' in unemployment_2020.columns:
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=unemployment_2020,
        x='region',
        y='estimated_unemployment_rate_(%)',
        hue='is_covid' if 'is_covid' in unemployment_2020.columns else None,
        palette='coolwarm',
        errorbar=None
    )
    plt.title('Regional Unemployment Rates' + 
             (' (COVID vs Non-COVID)' if 'is_covid' in unemployment_2020.columns else ''),
             pad=20)
    plt.xlabel('Region')
    plt.ylabel('Unemployment Rate (%)')
    plt.xticks(rotation=45)
    plt.legend(title='COVID Period' if 'is_covid' in unemployment_2020.columns else None)
    plt.tight_layout()
    plt.show()
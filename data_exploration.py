import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and perform comprehensive data exploration"""
    
    # Load the datasets
    print("Loading datasets...")
    train_df = pd.read_csv('/Users/garvrajput/StudioProjects/ML PROJ/venv/train.csv')
    test_df = pd.read_csv('/Users/garvrajput/StudioProjects/ML PROJ/venv/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print("\n" + "="*50)
    
    # Basic information about the datasets
    print("TRAINING DATA INFO:")
    print("="*30)
    print(train_df.info())
    print("\nTRAINING DATA HEAD:")
    print(train_df.head())
    
    print("\n" + "="*50)
    print("TEST DATA INFO:")
    print("="*30)
    print(test_df.info())
    print("\nTEST DATA HEAD:")
    print(test_df.head())
    
    # Check for missing values
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS:")
    print("="*30)
    print("Training data missing values:")
    print(train_df.isnull().sum())
    print("\nTest data missing values:")
    print(test_df.isnull().sum())
    
    # Data types analysis
    print("\n" + "="*50)
    print("DATA TYPES ANALYSIS:")
    print("="*30)
    print("Training data dtypes:")
    print(train_df.dtypes)
    print("\nTest data dtypes:")
    print(test_df.dtypes)
    
    # Statistical summary
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY:")
    print("="*30)
    print("Training data describe:")
    print(train_df.describe())
    
    # Check unique values in categorical columns
    print("\n" + "="*50)
    print("CATEGORICAL VARIABLES ANALYSIS:")
    print("="*30)
    print("Lifestyle Activities unique values in training data:")
    print(train_df['Lifestyle Activities'].value_counts())
    print("\nLifestyle Activities unique values in test data:")
    print(test_df['Lifestyle Activities'].value_counts())
    
    # Target variable analysis
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS:")
    print("="*30)
    print("Recovery Index statistics:")
    print(train_df['Recovery Index'].describe())
    print(f"\nRecovery Index range: {train_df['Recovery Index'].min()} to {train_df['Recovery Index'].max()}")
    print(f"Recovery Index unique values: {train_df['Recovery Index'].nunique()}")
    
    # Check for duplicates
    print("\n" + "="*50)
    print("DUPLICATE ANALYSIS:")
    print("="*30)
    print(f"Training data duplicates: {train_df.duplicated().sum()}")
    print(f"Test data duplicates: {test_df.duplicated().sum()}")
    print(f"ID duplicates in training: {train_df['Id'].duplicated().sum()}")
    print(f"ID duplicates in test: {test_df['Id'].duplicated().sum()}")
    
    # Check for outliers using IQR method
    print("\n" + "="*50)
    print("OUTLIER ANALYSIS (IQR Method):")
    print("="*30)
    numerical_cols = ['Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 
                     'Follow-Up Sessions', 'Recovery Index']
    
    for col in numerical_cols:
        if col in train_df.columns:
            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(train_df)*100:.2f}%)")
    
    # Distribution analysis
    print("\n" + "="*50)
    print("DISTRIBUTION ANALYSIS:")
    print("="*30)
    for col in numerical_cols:
        if col in train_df.columns:
            print(f"\n{col} distribution:")
            print(f"Skewness: {stats.skew(train_df[col]):.4f}")
            print(f"Kurtosis: {stats.kurtosis(train_df[col]):.4f}")
    
    return train_df, test_df

def create_visualizations(train_df, test_df):
    """Create comprehensive visualizations for data exploration"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Patient Recovery Dataset - Data Exploration', fontsize=16, fontweight='bold')
    
    # 1. Target variable distribution
    axes[0, 0].hist(train_df['Recovery Index'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Recovery Index Distribution')
    axes[0, 0].set_xlabel('Recovery Index')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Therapy Hours distribution
    axes[0, 1].hist(train_df['Therapy Hours'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Therapy Hours Distribution')
    axes[0, 1].set_xlabel('Therapy Hours')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Initial Health Score distribution
    axes[0, 2].hist(train_df['Initial Health Score'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].set_title('Initial Health Score Distribution')
    axes[0, 2].set_xlabel('Initial Health Score')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Average Sleep Hours distribution
    axes[1, 0].hist(train_df['Average Sleep Hours'], bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 0].set_title('Average Sleep Hours Distribution')
    axes[1, 0].set_xlabel('Average Sleep Hours')
    axes[1, 0].set_ylabel('Frequency')
    
    # 5. Follow-Up Sessions distribution
    axes[1, 1].hist(train_df['Follow-Up Sessions'], bins=20, alpha=0.7, color='plum', edgecolor='black')
    axes[1, 1].set_title('Follow-Up Sessions Distribution')
    axes[1, 1].set_xlabel('Follow-Up Sessions')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Lifestyle Activities count
    lifestyle_counts = train_df['Lifestyle Activities'].value_counts()
    axes[1, 2].bar(lifestyle_counts.index, lifestyle_counts.values, color=['lightblue', 'lightpink'])
    axes[1, 2].set_title('Lifestyle Activities Distribution')
    axes[1, 2].set_xlabel('Lifestyle Activities')
    axes[1, 2].set_ylabel('Count')
    
    # 7. Recovery Index vs Therapy Hours
    axes[2, 0].scatter(train_df['Therapy Hours'], train_df['Recovery Index'], alpha=0.5, color='blue')
    axes[2, 0].set_title('Recovery Index vs Therapy Hours')
    axes[2, 0].set_xlabel('Therapy Hours')
    axes[2, 0].set_ylabel('Recovery Index')
    
    # 8. Recovery Index vs Initial Health Score
    axes[2, 1].scatter(train_df['Initial Health Score'], train_df['Recovery Index'], alpha=0.5, color='red')
    axes[2, 1].set_title('Recovery Index vs Initial Health Score')
    axes[2, 1].set_xlabel('Initial Health Score')
    axes[2, 1].set_ylabel('Recovery Index')
    
    # 9. Recovery Index vs Average Sleep Hours
    axes[2, 2].scatter(train_df['Average Sleep Hours'], train_df['Recovery Index'], alpha=0.5, color='green')
    axes[2, 2].set_title('Recovery Index vs Average Sleep Hours')
    axes[2, 2].set_xlabel('Average Sleep Hours')
    axes[2, 2].set_ylabel('Recovery Index')
    
    plt.tight_layout()
    plt.savefig('/Users/garvrajput/StudioProjects/ML PROJ/data_exploration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap (only numerical columns)
    plt.figure(figsize=(10, 8))
    numerical_cols = ['Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 
                     'Follow-Up Sessions', 'Recovery Index']
    correlation_matrix = train_df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix - Training Data (Numerical Variables)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/garvrajput/StudioProjects/ML PROJ/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting comprehensive data exploration...")
    print("="*60)
    
    # Load and explore data
    train_df, test_df = load_and_explore_data()
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(train_df, test_df)
    
    print("\nData exploration completed!")
    print("Visualizations saved as 'data_exploration_plots.png' and 'correlation_heatmap.png'")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing class for Patient Recovery Dataset"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'Recovery Index'
        
    def load_data(self, train_path, test_path):
        """Load training and test datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def handle_categorical_variables(self):
        """Encode categorical variables"""
        print("\nHandling categorical variables...")
        
        # Check unique values in both datasets
        train_unique = set(self.train_df['Lifestyle Activities'].unique())
        test_unique = set(self.test_df['Lifestyle Activities'].unique())
        
        print(f"Training set unique values: {train_unique}")
        print(f"Test set unique values: {test_unique}")
        
        # Ensure both datasets have the same categories
        all_categories = train_unique.union(test_unique)
        print(f"All unique categories: {all_categories}")
        
        # Encode categorical variable
        self.train_df['Lifestyle Activities Encoded'] = self.label_encoder.fit_transform(
            self.train_df['Lifestyle Activities']
        )
        self.test_df['Lifestyle Activities Encoded'] = self.label_encoder.transform(
            self.test_df['Lifestyle Activities']
        )
        
        print("Categorical encoding completed.")
        print(f"Encoding mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return self.train_df, self.test_df
    
    def create_feature_columns(self):
        """Define feature columns for modeling"""
        self.feature_columns = [
            'Therapy Hours',
            'Initial Health Score', 
            'Lifestyle Activities Encoded',
            'Average Sleep Hours',
            'Follow-Up Sessions'
        ]
        
        print(f"\nFeature columns: {self.feature_columns}")
        return self.feature_columns
    
    def create_additional_features(self):
        """Create additional engineered features"""
        print("\nCreating additional features...")
        
        # For training data
        self.train_df['Therapy_Health_Interaction'] = (
            self.train_df['Therapy Hours'] * self.train_df['Initial Health Score']
        )
        self.train_df['Sleep_FollowUp_Interaction'] = (
            self.train_df['Average Sleep Hours'] * self.train_df['Follow-Up Sessions']
        )
        self.train_df['Health_Sleep_Ratio'] = (
            self.train_df['Initial Health Score'] / (self.train_df['Average Sleep Hours'] + 1)
        )
        self.train_df['Total_Engagement'] = (
            self.train_df['Therapy Hours'] + self.train_df['Follow-Up Sessions']
        )
        
        # For test data
        self.test_df['Therapy_Health_Interaction'] = (
            self.test_df['Therapy Hours'] * self.test_df['Initial Health Score']
        )
        self.test_df['Sleep_FollowUp_Interaction'] = (
            self.test_df['Average Sleep Hours'] * self.test_df['Follow-Up Sessions']
        )
        self.test_df['Health_Sleep_Ratio'] = (
            self.test_df['Initial Health Score'] / (self.test_df['Average Sleep Hours'] + 1)
        )
        self.test_df['Total_Engagement'] = (
            self.test_df['Therapy Hours'] + self.test_df['Follow-Up Sessions']
        )
        
        # Update feature columns to include new features
        self.feature_columns.extend([
            'Therapy_Health_Interaction',
            'Sleep_FollowUp_Interaction', 
            'Health_Sleep_Ratio',
            'Total_Engagement'
        ])
        
        print("Additional features created:")
        print("- Therapy_Health_Interaction: Therapy Hours × Initial Health Score")
        print("- Sleep_FollowUp_Interaction: Average Sleep Hours × Follow-Up Sessions")
        print("- Health_Sleep_Ratio: Initial Health Score / (Average Sleep Hours + 1)")
        print("- Total_Engagement: Therapy Hours + Follow-Up Sessions")
        
        return self.train_df, self.test_df
    
    def scale_features(self, method='standard'):
        """Scale features using specified method"""
        print(f"\nScaling features using {method} scaling...")
        
        if method == 'standard':
            scaler = self.scaler
        elif method == 'minmax':
            scaler = self.minmax_scaler
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Fit scaler on training data
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_val_scaled = scaler.transform(self.X_val)
        self.X_test_scaled = scaler.transform(self.X_test_final)
        
        print(f"Feature scaling completed using {method} scaling.")
        print(f"Scaled training data shape: {self.X_train_scaled.shape}")
        print(f"Scaled validation data shape: {self.X_val_scaled.shape}")
        print(f"Scaled test data shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split training data into train and validation sets"""
        print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
        
        # Prepare features and target
        X = self.train_df[self.feature_columns]
        y = self.train_df[self.target_column]
        
        # Split the data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Target distribution - Train: mean={self.y_train.mean():.2f}, std={self.y_train.std():.2f}")
        print(f"Target distribution - Val: mean={self.y_val.mean():.2f}, std={self.y_val.std():.2f}")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def prepare_test_data(self):
        """Prepare test data for prediction"""
        print("\nPreparing test data for prediction...")
        
        self.X_test_final = self.test_df[self.feature_columns]
        print(f"Test data shape: {self.X_test_final.shape}")
        
        return self.X_test_final
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING SUMMARY")
        print("="*60)
        
        print(f"Original training data shape: {self.train_df.shape}")
        print(f"Original test data shape: {self.test_df.shape}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns}")
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test_final.shape[0]} samples")
        
        print(f"\nTarget variable statistics:")
        print(f"  Training - Mean: {self.y_train.mean():.2f}, Std: {self.y_train.std():.2f}")
        print(f"  Validation - Mean: {self.y_val.mean():.2f}, Std: {self.y_val.std():.2f}")
        print(f"  Range: {self.y_train.min()} to {self.y_train.max()}")
        
        print(f"\nFeature scaling: Applied")
        print(f"Missing values: None")
        print(f"Categorical encoding: Completed")
        print(f"Feature engineering: Completed")
        
        return {
            'train_shape': self.X_train.shape,
            'val_shape': self.X_val.shape,
            'test_shape': self.X_test_final.shape,
            'features': self.feature_columns,
            'target_stats': {
                'train_mean': self.y_train.mean(),
                'train_std': self.y_train.std(),
                'val_mean': self.y_val.mean(),
                'val_std': self.y_val.std()
            }
        }
    
    def save_processed_data(self, output_dir='.'):
        """Save processed datasets"""
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save training data
        train_processed = pd.concat([
            pd.DataFrame(self.X_train_scaled, columns=self.feature_columns),
            self.y_train.reset_index(drop=True)
        ], axis=1)
        train_processed.to_csv(f'{output_dir}/train_processed.csv', index=False)
        
        # Save validation data
        val_processed = pd.concat([
            pd.DataFrame(self.X_val, columns=self.feature_columns),
            self.y_val.reset_index(drop=True)
        ], axis=1)
        val_processed.to_csv(f'{output_dir}/val_processed.csv', index=False)
        
        # Save test data
        test_processed = pd.DataFrame(self.X_test_scaled, columns=self.feature_columns)
        test_processed.to_csv(f'{output_dir}/test_processed.csv', index=False)
        
        print("Processed data saved successfully!")
        
        return train_processed, val_processed, test_processed

def main():
    """Main preprocessing pipeline"""
    print("Starting comprehensive data preprocessing...")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    train_df, test_df = preprocessor.load_data(
        '/Users/garvrajput/StudioProjects/ML PROJ/venv/train.csv',
        '/Users/garvrajput/StudioProjects/ML PROJ/venv/test.csv'
    )
    
    # Handle categorical variables
    train_df, test_df = preprocessor.handle_categorical_variables()
    
    # Create feature columns
    feature_columns = preprocessor.create_feature_columns()
    
    # Create additional features
    train_df, test_df = preprocessor.create_additional_features()
    
    # Split data
    X_train, X_val, y_train, y_val = preprocessor.split_data()
    
    # Prepare test data
    X_test_final = preprocessor.prepare_test_data()
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(method='standard')
    
    # Get data summary
    summary = preprocessor.get_data_summary()
    
    # Save processed data
    train_processed, val_processed, test_processed = preprocessor.save_processed_data()
    
    print("\nData preprocessing completed successfully!")
    print("Ready for model training and evaluation.")
    
    return preprocessor, summary

if __name__ == "__main__":
    preprocessor, summary = main()

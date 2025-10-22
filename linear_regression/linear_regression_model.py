import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionModel:
    """Linear Regression Model for Patient Recovery Prediction"""
    
    def __init__(self):
        self.model = None
        self.coefficients = None
        self.training_history = {}
        
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train linear regression model"""
        print("Training Linear Regression model...")
        print("="*50)
        
        # Create and train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Store results
        self.training_history = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_predictions': y_train_pred,
            'val_predictions': y_val_pred
        }
        
        print(f"Linear Regression Results:")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return self.model
    
    def analyze_coefficients(self, feature_names):
        """Analyze and visualize model coefficients"""
        print("\nAnalyzing model coefficients...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Get coefficients
        self.coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(f"Intercept: {self.model.intercept_:.4f}")
        print("\nCoefficients:")
        for idx, row in self.coefficients.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in self.coefficients['coefficient']]
        plt.barh(range(len(self.coefficients)), self.coefficients['coefficient'], color=colors)
        plt.yticks(range(len(self.coefficients)), self.coefficients['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Linear Regression - Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plt.savefig('./linear_regression/linear_regression_coefficients.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.coefficients
    
    def cross_validation_analysis(self, X_train, y_train, cv=5):
        """Perform cross-validation analysis"""
        print(f"\nPerforming {cv}-fold cross-validation...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, 
                                   scoring='neg_root_mean_squared_error')
        cv_rmse_scores = -cv_scores
        
        print(f"Cross-validation RMSE scores: {cv_rmse_scores}")
        print(f"Mean CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")
        
        # Store CV results
        self.training_history['cv_scores'] = cv_rmse_scores
        self.training_history['mean_cv_rmse'] = cv_rmse_scores.mean()
        self.training_history['std_cv_rmse'] = cv_rmse_scores.std()
        
        return cv_rmse_scores
    
    def create_visualizations(self, X_val, y_val, feature_names):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return
        
        # Get predictions
        y_val_pred = self.model.predict(X_val)
        residuals = y_val - y_val_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Linear Regression Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_val, y_val_pred, alpha=0.6, color='blue', s=30)
        axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Recovery Index')
        axes[0, 0].set_ylabel('Predicted Recovery Index')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(alpha=0.3)
        r2 = r2_score(y_val, y_val_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals plot
        axes[0, 1].scatter(y_val_pred, residuals, alpha=0.6, color='green', s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Recovery Index')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Residuals distribution
        axes[0, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].grid(alpha=0.3)
        
        # 4. Coefficients bar plot
        if self.coefficients is not None:
            colors = ['red' if x < 0 else 'green' for x in self.coefficients['coefficient']]
            axes[1, 0].barh(range(len(self.coefficients)), self.coefficients['coefficient'], color=colors)
            axes[1, 0].set_yticks(range(len(self.coefficients)))
            axes[1, 0].set_yticklabels(self.coefficients['feature'])
            axes[1, 0].set_xlabel('Coefficient Value')
            axes[1, 0].set_title('Feature Coefficients')
            axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            axes[1, 0].grid(alpha=0.3, axis='x')
        
        # 5. Prediction distribution
        axes[1, 1].hist(y_val, alpha=0.7, label='Actual', bins=30, color='blue', edgecolor='black')
        axes[1, 1].hist(y_val_pred, alpha=0.7, label='Predicted', bins=30, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Recovery Index')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 6. Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot (Residuals Normality)')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./linear_regression/linear_regression_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualizations saved")
    
    def predict_test_data(self, X_test, test_ids):
        """Make predictions on test data"""
        print("\nMaking predictions on test data...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Make predictions (unrounded)
        test_predictions = self.model.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'Id': test_ids,
            'Recovery Index': test_predictions
        })
        
        print(f"Test predictions shape: {test_predictions.shape}")
        print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
        print(f"Prediction mean: {test_predictions.mean():.4f}")
        print(f"Prediction std: {test_predictions.std():.4f}")
        
        # Save submission file
        submission.to_csv('./linear_regression/linear_regression_submission.csv', 
                         index=False)
        print("✓ Submission file saved: linear_regression_submission.csv")
        
        return submission
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        print("\n" + "="*60)
        print("LINEAR REGRESSION MODEL SUMMARY")
        print("="*60)
        
        if self.model is None:
            print("No model trained yet!")
            return
        
        print(f"\nModel Parameters:")
        print(f"  Intercept: {self.model.intercept_:.4f}")
        print(f"  Number of features: {len(self.model.coef_)}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Training RMSE: {self.training_history['train_rmse']:.4f}")
        print(f"  Validation RMSE: {self.training_history['val_rmse']:.4f}")
        print(f"  Training R²: {self.training_history['train_r2']:.4f}")
        print(f"  Validation R²: {self.training_history['val_r2']:.4f}")
        print(f"  Training MAE: {self.training_history['train_mae']:.4f}")
        print(f"  Validation MAE: {self.training_history['val_mae']:.4f}")
        
        if 'mean_cv_rmse' in self.training_history:
            print(f"\nCross-Validation:")
            print(f"  Mean CV RMSE: {self.training_history['mean_cv_rmse']:.4f}")
            print(f"  CV RMSE Std: {self.training_history['std_cv_rmse']:.4f}")
        
        print("="*60)
        return self.training_history

def main():
    """Main function to run Linear Regression model"""
    print("="*60)
    print("Starting Linear Regression Model Training...")
    print("="*60)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data(
        './data/train.csv',
        './data/test.csv'
    )
    
    # Preprocess data
    train_df, test_df = preprocessor.handle_categorical_variables()
    preprocessor.create_feature_columns()
    train_df, test_df = preprocessor.create_additional_features()
    X_train, X_val, y_train, y_val = preprocessor.split_data()
    X_test_final = preprocessor.prepare_test_data()
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(method='standard')
    
    # Get feature names and test IDs
    feature_names = preprocessor.feature_columns
    test_ids = test_df['Id'].values
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {feature_names}")
    
    # Initialize and train model
    lr_model = LinearRegressionModel()
    lr_model.train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Analyze coefficients
    lr_model.analyze_coefficients(feature_names)
    
    # Cross-validation analysis
    lr_model.cross_validation_analysis(X_train_scaled, y_train)
    
    # Create visualizations
    lr_model.create_visualizations(X_val_scaled, y_val, feature_names)
    
    # Make test predictions
    submission = lr_model.predict_test_data(X_test_scaled, test_ids)
    
    # Get model summary
    summary = lr_model.get_model_summary()
    
    print("\n" + "="*60)
    print("Linear Regression model training completed!")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ linear_regression_coefficients.png")
    print("  ✓ linear_regression_analysis.png")
    print("  ✓ linear_regression_submission.csv")
    
    return lr_model, summary

if __name__ == "__main__":
    model, summary = main()

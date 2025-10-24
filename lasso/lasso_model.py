import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class LassoRegressionModel:
    """Lasso Regression Model for Patient Recovery Prediction"""
    
    def __init__(self):
        self.model = None
        self.best_alpha = None
        self.coefficients = None
        self.training_history = {}
        
    def train_basic_model(self, X_train, y_train, X_val, y_val, alpha=0.01):
        """Train basic lasso regression model"""
        print(f"Training Lasso Regression model (alpha={alpha})...")
        print("="*50)
        
        self.model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        self.model.fit(X_train, y_train)
        
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        self.training_history['basic'] = {
            'alpha': alpha,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_predictions': y_train_pred,
            'val_predictions': y_val_pred
        }
        
        print(f"Lasso Results (alpha={alpha}):")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter tuning for alpha"""
        print("\nPerforming hyperparameter tuning...")
        print("="*50)
        
        alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        print(f"Testing alphas: {alphas}")
        
        lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
        lasso_cv.fit(X_train, y_train)
        
        self.best_alpha = lasso_cv.alpha_
        print(f"\nBest alpha: {self.best_alpha}")
        
        self.model = Lasso(alpha=self.best_alpha, random_state=42, max_iter=10000)
        self.model.fit(X_train, y_train)
        
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        self.training_history['tuned'] = {
            'best_alpha': self.best_alpha,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_predictions': y_train_pred,
            'val_predictions': y_val_pred
        }
        
        print(f"\nTuned Lasso Results:")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Non-zero coefficients: {np.sum(self.model.coef_ != 0)}/{len(self.model.coef_)}")
        
        # Test all alphas
        self.alpha_comparison = []
        for alpha in alphas:
            model_temp = Lasso(alpha=alpha, max_iter=10000, random_state=42)
            model_temp.fit(X_train, y_train)
            val_pred = model_temp.predict(X_val)
            val_rmse_temp = np.sqrt(mean_squared_error(y_val, val_pred))
            non_zero = np.sum(model_temp.coef_ != 0)
            self.alpha_comparison.append({
                'alpha': alpha, 
                'val_rmse': val_rmse_temp,
                'non_zero_coefs': non_zero
            })
        
        return self.model
    
    def analyze_coefficients(self, feature_names):
        """Analyze and visualize model coefficients with feature selection"""
        print("\nAnalyzing model coefficients and feature selection...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        self.coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        non_zero_features = self.coefficients[self.coefficients['coefficient'] != 0]
        zero_features = self.coefficients[self.coefficients['coefficient'] == 0]
        
        print(f"Intercept: {self.model.intercept_:.4f}")
        print(f"Alpha: {self.best_alpha if self.best_alpha else 'Not tuned'}")
        print(f"\nFeature Selection:")
        print(f"  Non-zero features: {len(non_zero_features)}")
        print(f"  Zero features (eliminated): {len(zero_features)}")
        
        print("\nNon-zero Coefficients:")
        for idx, row in non_zero_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        if len(zero_features) > 0:
            print("\nEliminated Features:")
            for idx, row in zero_features.iterrows():
                print(f"  {row['feature']}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Coefficients
        colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' 
                  for x in self.coefficients['coefficient']]
        axes[0].barh(range(len(self.coefficients)), self.coefficients['coefficient'], color=colors)
        axes[0].set_yticks(range(len(self.coefficients)))
        axes[0].set_yticklabels(self.coefficients['feature'])
        axes[0].set_xlabel('Coefficient Value')
        axes[0].set_title(f'Lasso Coefficients (α={self.best_alpha if self.best_alpha else "default"})')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0].grid(alpha=0.3, axis='x')
        
        # Alpha vs Features plot
        if hasattr(self, 'alpha_comparison'):
            alpha_df = pd.DataFrame(self.alpha_comparison)
            ax2 = axes[1]
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(alpha_df['alpha'], alpha_df['val_rmse'], 
                            marker='o', color='blue', label='Val RMSE', linewidth=2)
            line2 = ax2_twin.plot(alpha_df['alpha'], alpha_df['non_zero_coefs'], 
                                 marker='s', color='red', label='Non-zero Coefs', linewidth=2)
            
            ax2.set_xscale('log')
            ax2.set_xlabel('Alpha (log scale)')
            ax2.set_ylabel('Validation RMSE', color='blue')
            ax2_twin.set_ylabel('Non-zero Coefficients', color='red')
            ax2.set_title('Lasso Regularization Path')
            ax2.axvline(x=self.best_alpha, color='green', linestyle='--', 
                       label=f'Best α={self.best_alpha}', linewidth=2)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='best')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./lasso/lasso_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.coefficients
    
    def cross_validation_analysis(self, X_train, y_train, cv=5):
        """Perform cross-validation analysis"""
        print(f"\nPerforming {cv}-fold cross-validation...")
        print("="*50)
        
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, 
                                   scoring='neg_root_mean_squared_error')
        cv_rmse_scores = -cv_scores
        
        print(f"Mean CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")
        
        self.training_history['cv_scores'] = cv_rmse_scores
        self.training_history['mean_cv_rmse'] = cv_rmse_scores.mean()
        
        return cv_rmse_scores
    
    def create_visualizations(self, X_val, y_val, feature_names):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        print("="*50)
        
        y_val_pred = self.model.predict(X_val)
        residuals = y_val - y_val_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Lasso Regression Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_val, y_val_pred, alpha=0.6, color='blue', s=30)
        axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Residuals
        axes[0, 1].scatter(y_val_pred, residuals, alpha=0.6, color='green', s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Residuals distribution
        axes[0, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].grid(alpha=0.3)
        
        # 4. Coefficients
        non_zero_coefs = self.coefficients[self.coefficients['coefficient'] != 0]
        if len(non_zero_coefs) > 0:
            colors = ['red' if x < 0 else 'green' for x in non_zero_coefs['coefficient']]
            axes[1, 0].barh(range(len(non_zero_coefs)), non_zero_coefs['coefficient'], color=colors)
            axes[1, 0].set_yticks(range(len(non_zero_coefs)))
            axes[1, 0].set_yticklabels(non_zero_coefs['feature'])
            axes[1, 0].set_title('Non-zero Coefficients')
            axes[1, 0].axvline(x=0, color='black', linewidth=0.8)
            axes[1, 0].grid(alpha=0.3, axis='x')
        
        # 5. Prediction distribution
        axes[1, 1].hist(y_val, alpha=0.7, label='Actual', bins=30, color='blue')
        axes[1, 1].hist(y_val_pred, alpha=0.7, label='Predicted', bins=30, color='red')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 6. Feature selection visualization
        if hasattr(self, 'alpha_comparison'):
            alpha_df = pd.DataFrame(self.alpha_comparison)
            axes[1, 2].plot(alpha_df['alpha'], alpha_df['non_zero_coefs'], 
                           marker='o', linewidth=2, markersize=8)
            axes[1, 2].set_xscale('log')
            axes[1, 2].set_xlabel('Alpha (log scale)')
            axes[1, 2].set_ylabel('Number of Features')
            axes[1, 2].set_title('Feature Selection Path')
            axes[1, 2].axvline(x=self.best_alpha, color='r', linestyle='--')
            axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./lasso/lasso_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualizations saved")
    
    def predict_test_data(self, X_test, test_ids):
        """Make predictions on test data"""
        print("\nMaking predictions on test data...")
        print("="*50)
        
        test_predictions = self.model.predict(X_test)
        
        submission = pd.DataFrame({
            'Id': test_ids,
            'Recovery Index': test_predictions
        })
        
        print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
        
        submission.to_csv('./lasso/lasso_submission.csv', index=False)
        print("✓ Submission saved: lasso_submission.csv")
        
        return submission
    
    def get_model_summary(self):
        """Get model summary"""
        print("\n" + "="*60)
        print("LASSO REGRESSION MODEL SUMMARY")
        print("="*60)
        
        print(f"\nModel Parameters:")
        print(f"  Alpha: {self.best_alpha if self.best_alpha else 'default'}")
        print(f"  Non-zero features: {np.sum(self.model.coef_ != 0)}/{len(self.model.coef_)}")
        
        if 'tuned' in self.training_history:
            r = self.training_history['tuned']
            print(f"\nPerformance:")
            print(f"  Training RMSE: {r['train_rmse']:.4f}")
            print(f"  Validation RMSE: {r['val_rmse']:.4f}")
            print(f"  Training R²: {r['train_r2']:.4f}")
            print(f"  Validation R²: {r['val_r2']:.4f}")
        
        print("="*60)
        return self.training_history

def main():
    """Main function"""
    print("="*60)
    print("Starting Lasso Regression Model Training...")
    print("="*60)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data(
        './data/train_cleaned.csv',
        './data/test_cleaned.csv'
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
    
    lasso_model = LassoRegressionModel()
    lasso_model.train_basic_model(X_train_scaled, y_train, X_val_scaled, y_val)
    lasso_model.hyperparameter_tuning(X_train_scaled, y_train, X_val_scaled, y_val)
    lasso_model.analyze_coefficients(feature_names)
    lasso_model.cross_validation_analysis(X_train_scaled, y_train)
    lasso_model.create_visualizations(X_val_scaled, y_val, feature_names)
    lasso_model.predict_test_data(X_test_scaled, test_ids)
    lasso_model.get_model_summary()
    
    print("\n" + "="*60)
    print("Lasso model training completed!")
    print("Generated files:")
    print("  ✓ lasso_coefficients.png")
    print("  ✓ lasso_analysis.png")
    print("  ✓ lasso_submission.csv")
    
    return lasso_model

if __name__ == "__main__":
    model = main()

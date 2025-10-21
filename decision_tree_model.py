import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeModel:
    """Decision Tree Regression Model for Patient Recovery Prediction"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_history = {}
        
    def train_basic_model(self, X_train, y_train, X_val, y_val):
        """Train a basic decision tree model"""
        print("Training basic Decision Tree model...")
        print("="*50)
        
        # Create and train basic model
        self.model = DecisionTreeRegressor(random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Store results
        self.training_history['basic'] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_predictions': y_train_pred,
            'val_predictions': y_val_pred
        }
        
        print(f"Basic Decision Tree Results:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("\nPerforming hyperparameter tuning...")
        print("="*50)
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 5, 10, 20],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
        }
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        print("\nStarting grid search...")
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        self.best_params = grid_search.best_params_
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Train model with best parameters
        self.model = grid_search.best_estimator_
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Store results
        self.training_history['tuned'] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_predictions': y_train_pred,
            'val_predictions': y_val_pred,
            'best_params': self.best_params
        }
        
        print(f"\nTuned Decision Tree Results:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return self.model
    
    def analyze_feature_importance(self, feature_names):
        """Analyze and visualize feature importance"""
        print("\nAnalyzing feature importance...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance:")
        for idx, row in self.feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.feature_importance, x='importance', y='feature')
        plt.title('Decision Tree - Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('/Users/garvrajput/StudioProjects/ML PROJ/decision_tree_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.feature_importance
    
    def cross_validation_analysis(self, X_train, y_train, cv=5):
        """Perform cross-validation analysis"""
        print(f"\nPerforming {cv}-fold cross-validation...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, 
                                  scoring='neg_mean_squared_error')
        cv_scores = -cv_scores  # Convert to positive MSE
        
        print(f"Cross-validation MSE scores: {cv_scores}")
        print(f"Mean CV MSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store CV results
        self.training_history['cv'] = {
            'cv_scores': cv_scores,
            'mean_cv_mse': cv_scores.mean(),
            'std_cv_mse': cv_scores.std()
        }
        
        return cv_scores
    
    def create_visualizations(self, X_val, y_val, feature_names):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return
        
        # Get predictions
        y_val_pred = self.model.predict(X_val)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Decision Tree Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_val, y_val_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Recovery Index')
        axes[0, 0].set_ylabel('Predicted Recovery Index')
        axes[0, 0].set_title('Actual vs Predicted (Validation Set)')
        
        # Add R² score to plot
        r2 = r2_score(y_val, y_val_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals plot
        residuals = y_val - y_val_pred
        axes[0, 1].scatter(y_val_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Recovery Index')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top Feature Importance')
        
        # 4. Prediction distribution
        axes[1, 1].hist(y_val, alpha=0.7, label='Actual', bins=30, color='blue')
        axes[1, 1].hist(y_val_pred, alpha=0.7, label='Predicted', bins=30, color='red')
        axes[1, 1].set_xlabel('Recovery Index')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/Users/garvrajput/StudioProjects/ML PROJ/decision_tree_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_test_data(self, X_test, test_ids):
        """Make predictions on test data"""
        print("\nMaking predictions on test data...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Make predictions
        test_predictions = self.model.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'Id': test_ids,
            'Recovery Index': test_predictions
        })
        
        print(f"Test predictions shape: {test_predictions.shape}")
        print(f"Prediction range: {test_predictions.min():.2f} to {test_predictions.max():.2f}")
        print(f"Prediction mean: {test_predictions.mean():.2f}")
        
        # Save submission file
        submission.to_csv('/Users/garvrajput/StudioProjects/ML PROJ/decision_tree_submission.csv', 
                         index=False)
        print("Submission file saved as 'decision_tree_submission.csv'")
        
        return submission
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        print("\n" + "="*60)
        print("DECISION TREE MODEL SUMMARY")
        print("="*60)
        
        if self.model is None:
            print("No model trained yet!")
            return
        
        print(f"Model type: {type(self.model).__name__}")
        print(f"Best parameters: {self.best_params}")
        print(f"Number of features: {self.model.n_features_in_}")
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")
        
        if 'tuned' in self.training_history:
            results = self.training_history['tuned']
            print(f"\nPerformance Metrics:")
            print(f"  Training MSE: {results['train_mse']:.4f}")
            print(f"  Validation MSE: {results['val_mse']:.4f}")
            print(f"  Training R²: {results['train_r2']:.4f}")
            print(f"  Validation R²: {results['val_r2']:.4f}")
            print(f"  Training MAE: {results['train_mae']:.4f}")
            print(f"  Validation MAE: {results['val_mae']:.4f}")
        
        if 'cv' in self.training_history:
            cv_results = self.training_history['cv']
            print(f"\nCross-Validation:")
            print(f"  Mean CV MSE: {cv_results['mean_cv_mse']:.4f}")
            print(f"  CV MSE Std: {cv_results['std_cv_mse']:.4f}")
        
        return self.training_history

def main():
    """Main function to run Decision Tree model"""
    print("Starting Decision Tree Model Training...")
    print("="*60)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data(
        '/Users/garvrajput/StudioProjects/ML PROJ/venv/train.csv',
        '/Users/garvrajput/StudioProjects/ML PROJ/venv/test.csv'
    )
    
    # Preprocess data
    train_df, test_df = preprocessor.handle_categorical_variables()
    preprocessor.create_feature_columns()
    train_df, test_df = preprocessor.create_additional_features()
    X_train, X_val, y_train, y_val = preprocessor.split_data()
    X_test_final = preprocessor.prepare_test_data()
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(method='standard')
    
    # Get feature names
    feature_names = preprocessor.feature_columns
    
    # Initialize and train Decision Tree model
    dt_model = DecisionTreeModel()
    
    # Train basic model
    dt_model.train_basic_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Hyperparameter tuning
    dt_model.hyperparameter_tuning(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Analyze feature importance
    dt_model.analyze_feature_importance(feature_names)
    
    # Cross-validation analysis
    dt_model.cross_validation_analysis(X_train_scaled, y_train)
    
    # Create visualizations
    dt_model.create_visualizations(X_val_scaled, y_val, feature_names)
    
    # Make test predictions
    test_ids = test_df['Id'].values
    submission = dt_model.predict_test_data(X_test_scaled, test_ids)
    
    # Get model summary
    summary = dt_model.get_model_summary()
    
    print("\nDecision Tree model training completed!")
    print("Check the generated files:")
    print("- decision_tree_feature_importance.png")
    print("- decision_tree_analysis.png")
    print("- decision_tree_submission.csv")
    
    return dt_model, summary

if __name__ == "__main__":
    model, summary = main()

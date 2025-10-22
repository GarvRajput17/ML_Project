import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class RigorousXGBoostModel:
    """Rigorous XGBoost Regression Model for Patient Recovery Prediction - Kaggle/Colab Optimized"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_history = {}
        self.cv_results = {}
        
    def train_basic_model(self, X_train, y_train, X_val, y_val):
        """Train a basic XGBoost model with good defaults"""
        print("Training basic XGBoost model...")
        print("="*50)
        
        # Create and train basic model with optimized defaults
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            gamma=0,
            min_child_weight=1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
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
        
        print(f"Basic XGBoost Results:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return self.model
    
    def comprehensive_hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform comprehensive hyperparameter tuning with multiple strategies"""
        print("\nPerforming comprehensive XGBoost hyperparameter tuning...")
        print("="*60)
        
        # Strategy 1: Coarse Grid Search (fast exploration)
        print("\n1. COARSE GRID SEARCH (Fast Exploration)")
        print("-" * 40)
        
        coarse_param_grid = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        
        print("Coarse parameter grid:")
        for param, values in coarse_param_grid.items():
            print(f"  {param}: {values}")
        
        # Calculate total combinations
        total_combinations = 1
        for values in coarse_param_grid.values():
            total_combinations *= len(values)
        print(f"Total combinations: {total_combinations}")
        print(f"With 5-fold CV: {total_combinations * 5} total fits")
        
        # Coarse grid search
        coarse_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            coarse_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nStarting coarse grid search...")
        coarse_grid.fit(X_train, y_train)
        
        print(f"Best coarse parameters: {coarse_grid.best_params_}")
        print(f"Best coarse CV score: {-coarse_grid.best_score_:.4f}")
        
        # Strategy 2: Fine Grid Search around best parameters
        print("\n2. FINE GRID SEARCH (Refinement)")
        print("-" * 40)
        
        best_params = coarse_grid.best_params_
        
        # Create fine grid around best parameters
        fine_param_grid = {
            'n_estimators': [best_params['n_estimators'] - 200, best_params['n_estimators'], best_params['n_estimators'] + 200],
            'max_depth': [best_params['max_depth'] - 1, best_params['max_depth'], best_params['max_depth'] + 1],
            'learning_rate': [best_params['learning_rate'] * 0.5, best_params['learning_rate'], best_params['learning_rate'] * 1.5],
            'subsample': [best_params['subsample'] - 0.05, best_params['subsample'], best_params['subsample'] + 0.05],
            'colsample_bytree': [best_params['colsample_bytree'] - 0.05, best_params['colsample_bytree'], best_params['colsample_bytree'] + 0.05],
            'reg_alpha': [best_params['reg_alpha'], best_params['reg_alpha'] * 2, best_params['reg_alpha'] * 5],
            'reg_lambda': [best_params['reg_lambda'], best_params['reg_lambda'] * 2, best_params['reg_lambda'] * 5]
        }
        
        # Ensure values are within valid ranges
        for param, values in fine_param_grid.items():
            if param == 'n_estimators':
                fine_param_grid[param] = [max(100, min(2000, v)) for v in values]
            elif param in ['subsample', 'colsample_bytree']:
                fine_param_grid[param] = [max(0.1, min(1.0, v)) for v in values]
            elif param == 'learning_rate':
                fine_param_grid[param] = [max(0.001, min(0.3, v)) for v in values]
            elif param == 'max_depth':
                fine_param_grid[param] = [max(1, min(10, v)) for v in values]
        
        print("Fine parameter grid:")
        for param, values in fine_param_grid.items():
            print(f"  {param}: {values}")
        
        # Fine grid search
        fine_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            fine_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nStarting fine grid search...")
        fine_grid.fit(X_train, y_train)
        
        print(f"Best fine parameters: {fine_grid.best_params_}")
        print(f"Best fine CV score: {-fine_grid.best_score_:.4f}")
        
        # Strategy 3: Randomized Search for additional exploration
        print("\n3. RANDOMIZED SEARCH (Additional Exploration)")
        print("-" * 40)
        
        random_param_dist = {
            'n_estimators': [500, 800, 1000, 1200, 1500, 2000],
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bynode': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5],
            'gamma': [0, 0.1, 0.5, 1, 2],
            'min_child_weight': [1, 3, 5, 7, 10]
        }
        
        print("Randomized search parameter distribution:")
        for param, values in random_param_dist.items():
            print(f"  {param}: {values}")
        
        # Randomized search
        random_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            random_param_dist,
            n_iter=100,  # 100 random combinations
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        print("\nStarting randomized search (100 iterations)...")
        random_search.fit(X_train, y_train)
        
        print(f"Best random parameters: {random_search.best_params_}")
        print(f"Best random CV score: {-random_search.best_score_:.4f}")
        
        # Compare all three strategies
        print("\n4. STRATEGY COMPARISON")
        print("-" * 40)
        
        strategies = {
            'Coarse Grid': coarse_grid,
            'Fine Grid': fine_grid,
            'Random Search': random_search
        }
        
        best_strategy = None
        best_score = float('inf')
        
        for name, search in strategies.items():
            score = -search.best_score_
            print(f"{name}: CV Score = {score:.4f}")
            if score < best_score:
                best_score = score
                best_strategy = search
        
        print(f"\nBest strategy: {best_strategy}")
        print(f"Best overall CV score: {best_score:.4f}")
        
        # Use best model
        self.model = best_strategy.best_estimator_
        self.best_params = best_strategy.best_params_
        
        # Make predictions with best model
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
            'best_params': self.best_params,
            'best_cv_score': best_score
        }
        
        print(f"\nFinal Tuned XGBoost Results:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return self.model
    
    def advanced_cross_validation(self, X_train, y_train, cv_folds=10):
        """Perform advanced cross-validation analysis"""
        print(f"\nPerforming advanced {cv_folds}-fold cross-validation...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Create KFold object
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=kfold, 
                                  scoring='neg_mean_squared_error')
        cv_scores = -cv_scores  # Convert to positive MSE
        
        print(f"Cross-validation MSE scores: {cv_scores}")
        print(f"Mean CV MSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"CV MSE Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
        
        # Store CV results
        self.cv_results = {
            'cv_scores': cv_scores,
            'mean_cv_mse': cv_scores.mean(),
            'std_cv_mse': cv_scores.std(),
            'min_cv_mse': cv_scores.min(),
            'max_cv_mse': cv_scores.max()
        }
        
        # Create CV visualization
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, cv_folds + 1), cv_scores, 'bo-', alpha=0.7)
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
        plt.fill_between(range(1, cv_folds + 1), 
                        cv_scores.mean() - cv_scores.std(), 
                        cv_scores.mean() + cv_scores.std(), 
                        alpha=0.2, color='blue', label=f'±1 std: {cv_scores.std():.4f}')
        plt.xlabel('CV Fold')
        plt.ylabel('MSE')
        plt.title(f'XGBoost Cross-Validation Results ({cv_folds}-fold)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./xg_boost/xgboost_cv_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return cv_scores
    
    def analyze_feature_importance(self, feature_names):
        """Analyze and visualize feature importance with multiple methods"""
        print("\nAnalyzing XGBoost feature importance...")
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
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('XGBoost Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar plot
        top_features = self.feature_importance.head(10)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].set_title('Top 10 Feature Importance')
        
        # 2. Pie chart for top 5 features
        top_5 = self.feature_importance.head(5)
        axes[0, 1].pie(top_5['importance'], labels=top_5['feature'], autopct='%1.1f%%')
        axes[0, 1].set_title('Top 5 Features Distribution')
        
        # 3. Cumulative importance
        cumulative_importance = self.feature_importance['importance'].cumsum()
        axes[1, 0].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'bo-')
        axes[1, 0].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        axes[1, 0].axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        axes[1, 0].set_xlabel('Number of Features')
        axes[1, 0].set_ylabel('Cumulative Importance')
        axes[1, 0].set_title('Cumulative Feature Importance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Importance distribution
        axes[1, 1].hist(self.feature_importance['importance'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Feature Importance Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./xg_boost/xgboost_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.feature_importance
    
    def create_comprehensive_visualizations(self, X_val, y_val, feature_names):
        """Create comprehensive visualizations"""
        print("\nCreating comprehensive XGBoost visualizations...")
        print("="*50)
        
        if self.model is None:
            print("No model trained yet!")
            return
        
        # Get predictions
        y_val_pred = self.model.predict(X_val)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive XGBoost Model Analysis', fontsize=16, fontweight='bold')
        
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
        
        # 3. Residuals distribution
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].axvline(x=0, color='r', linestyle='--')
        
        # 4. Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top Feature Importance')
        
        # 5. Prediction distribution
        axes[1, 1].hist(y_val, alpha=0.7, label='Actual', bins=30, color='blue')
        axes[1, 1].hist(y_val_pred, alpha=0.7, label='Predicted', bins=30, color='red')
        axes[1, 1].set_xlabel('Recovery Index')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        
        # 6. Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot of Residuals')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Learning curve (if available)
        if hasattr(self.model, 'evals_result_'):
            axes[2, 0].plot(self.model.evals_result_['validation_0']['rmse'], label='Training')
            axes[2, 0].plot(self.model.evals_result_['validation_1']['rmse'], label='Validation')
            axes[2, 0].set_xlabel('Boosting Rounds')
            axes[2, 0].set_ylabel('RMSE')
            axes[2, 0].set_title('Learning Curve')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Learning Curve\nNot Available', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Learning Curve')
        
        # 8. Error analysis
        error_analysis = pd.DataFrame({
            'Actual': y_val,
            'Predicted': y_val_pred,
            'Error': residuals,
            'Abs_Error': np.abs(residuals)
        })
        
        axes[2, 1].scatter(error_analysis['Actual'], error_analysis['Abs_Error'], alpha=0.6)
        axes[2, 1].set_xlabel('Actual Recovery Index')
        axes[2, 1].set_ylabel('Absolute Error')
        axes[2, 1].set_title('Error Analysis')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Model performance metrics
        mse = mean_squared_error(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mse)
        
        metrics_text = f"""
        MSE: {mse:.4f}
        RMSE: {rmse:.4f}
        MAE: {mae:.4f}
        R²: {r2:.4f}
        """
        
        axes[2, 2].text(0.1, 0.5, metrics_text, transform=axes[2, 2].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[2, 2].set_title('Model Performance Metrics')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('./xg_boost/xgboost_comprehensive_analysis.png', 
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
        print(f"Prediction std: {test_predictions.std():.2f}")
        
        # Save submission file
        submission.to_csv('./xg_boost/xgboost_rigorous_submission.csv', 
                         index=False)
        print("Submission file saved as 'xgboost_rigorous_submission.csv'")
        
        return submission
    
    def get_comprehensive_summary(self):
        """Get comprehensive model summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE XGBOOST MODEL SUMMARY")
        print("="*80)
        
        if self.model is None:
            print("No model trained yet!")
            return
        
        print(f"Model type: {type(self.model).__name__}")
        print(f"Best parameters: {self.best_params}")
        print(f"Number of features: {self.model.n_features_in_}")
        print(f"Number of estimators: {self.model.n_estimators}")
        print(f"Learning rate: {self.model.learning_rate}")
        print(f"Max depth: {self.model.max_depth}")
        print(f"Subsample: {self.model.subsample}")
        print(f"Colsample bytree: {self.model.colsample_bytree}")
        print(f"Reg alpha: {self.model.reg_alpha}")
        print(f"Reg lambda: {self.model.reg_lambda}")
        
        if 'tuned' in self.training_history:
            results = self.training_history['tuned']
            print(f"\nPerformance Metrics:")
            print(f"  Training MSE: {results['train_mse']:.4f}")
            print(f"  Validation MSE: {results['val_mse']:.4f}")
            print(f"  Training R²: {results['train_r2']:.4f}")
            print(f"  Validation R²: {results['val_r2']:.4f}")
            print(f"  Training MAE: {results['train_mae']:.4f}")
            print(f"  Validation MAE: {results['val_mae']:.4f}")
            print(f"  Best CV Score: {results['best_cv_score']:.4f}")
        
        if self.cv_results:
            print(f"\nCross-Validation Results:")
            print(f"  Mean CV MSE: {self.cv_results['mean_cv_mse']:.4f}")
            print(f"  CV MSE Std: {self.cv_results['std_cv_mse']:.4f}")
            print(f"  Min CV MSE: {self.cv_results['min_cv_mse']:.4f}")
            print(f"  Max CV MSE: {self.cv_results['max_cv_mse']:.4f}")
        
        return self.training_history

def main():
    """Main function to run rigorous XGBoost model"""
    print("Starting Rigorous XGBoost Model Training (Kaggle/Colab Optimized)...")
    print("="*80)
    
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
    
    # Get feature names
    feature_names = preprocessor.feature_columns
    
    # Initialize and train rigorous XGBoost model
    xgboost_model = RigorousXGBoostModel()
    
    # Train basic model
    xgboost_model.train_basic_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Comprehensive hyperparameter tuning
    xgboost_model.comprehensive_hyperparameter_tuning(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Advanced cross-validation
    xgboost_model.advanced_cross_validation(X_train_scaled, y_train, cv_folds=10)
    
    # Analyze feature importance
    xgboost_model.analyze_feature_importance(feature_names)
    
    # Create comprehensive visualizations
    xgboost_model.create_comprehensive_visualizations(X_val_scaled, y_val, feature_names)
    
    # Make test predictions
    test_ids = test_df['Id'].values
    submission = xgboost_model.predict_test_data(X_test_scaled, test_ids)
    
    # Get comprehensive summary
    summary = xgboost_model.get_comprehensive_summary()
    
    print("\nRigorous XGBoost model training completed!")
    print("Check the generated files:")
    print("- xgboost_cv_results.png")
    print("- xgboost_feature_importance.png")
    print("- xgboost_comprehensive_analysis.png")
    print("- xgboost_rigorous_submission.csv")
    
    return xgboost_model, summary

if __name__ == "__main__":
    model, summary = main()

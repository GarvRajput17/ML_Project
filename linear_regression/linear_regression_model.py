import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, cross_validate
from data.data_preprocessing import DataPreprocessor


def train_and_evaluate_linear_regression():
    print("Starting Linear Regression baseline...")
    print("=" * 80)

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    # In data_preprocessing.py, change the load paths:
    train_df, test_df = preprocessor.load_data(
        './data/train_cleaned.csv',
        './data/test_cleaned.csv'
    )

    train_df, test_df = preprocessor.handle_categorical_variables()
    preprocessor.create_feature_columns()
    train_df, test_df = preprocessor.create_additional_features()
    X_train, X_val, y_train, y_val = preprocessor.split_data(test_size=0.20)
    X_test_final = preprocessor.prepare_test_data()

    # Linear models benefit from scaling
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(method='standard')

    # Train linear regression
    model = LinearRegression(n_jobs=None)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print("Linear Regression Results:")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Training R\u00b2: {train_r2:.4f}")
    print(f"Validation R\u00b2: {val_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")

    # K-Fold Cross-Validation
    print("\n" + "="*50)
    print("K-FOLD CROSS-VALIDATION ANALYSIS")
    print("="*50)
    
    # 5-Fold CV
    kfold_5 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_5fold = cross_validate(model, X_train_scaled, y_train, 
                                   cv=kfold_5, scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
                                   return_train_score=True)
    
    print("5-Fold Cross-Validation Results:")
    print(f"CV MSE (mean ± std): {-cv_scores_5fold['test_neg_mean_squared_error'].mean():.4f} ± {cv_scores_5fold['test_neg_mean_squared_error'].std():.4f}")
    print(f"CV R² (mean ± std): {cv_scores_5fold['test_r2'].mean():.4f} ± {cv_scores_5fold['test_r2'].std():.4f}")
    print(f"CV MAE (mean ± std): {-cv_scores_5fold['test_neg_mean_absolute_error'].mean():.4f} ± {cv_scores_5fold['test_neg_mean_absolute_error'].std():.4f}")
    
    # 10-Fold CV
    kfold_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores_10fold = cross_validate(model, X_train_scaled, y_train, 
                                    cv=kfold_10, scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
                                    return_train_score=True)
    
    print("\n10-Fold Cross-Validation Results:")
    print(f"CV MSE (mean ± std): {-cv_scores_10fold['test_neg_mean_squared_error'].mean():.4f} ± {cv_scores_10fold['test_neg_mean_squared_error'].std():.4f}")
    print(f"CV R² (mean ± std): {cv_scores_10fold['test_r2'].mean():.4f} ± {cv_scores_10fold['test_r2'].std():.4f}")
    print(f"CV MAE (mean ± std): {-cv_scores_10fold['test_neg_mean_absolute_error'].mean():.4f} ± {cv_scores_10fold['test_neg_mean_absolute_error'].std():.4f}")

    

    # Model Selection: Choose the best performing model
    print("\n" + "="*50)
    print("MODEL SELECTION & FINAL PREDICTIONS")
    print("="*50)
    
    # Compare validation performance with CV performance
    val_performance = val_mse
    cv_5fold_performance = -cv_scores_5fold['test_neg_mean_squared_error'].mean()
    cv_10fold_performance = -cv_scores_10fold['test_neg_mean_squared_error'].mean()
    
    print("Performance Comparison:")
    print(f"Validation MSE: {val_performance:.4f}")
    print(f"5-Fold CV MSE: {cv_5fold_performance:.4f}")
    print(f"10-Fold CV MSE: {cv_10fold_performance:.4f}")
    
    # Select best model based on validation performance
    # Since we only have one model, we'll use it, but we could retrain with best parameters
    best_model = model
    best_performance = val_performance
    best_method = "Validation Set"
    
    print(f"\nBest model selected: {best_method} (MSE: {best_performance:.4f})")
    print("Using the trained linear regression model for final predictions...")

    # Prepare Kaggle submission with best model
    test_ids = test_df['Id'].values
    test_predictions = best_model.predict(X_test_scaled)
    submission = pd.DataFrame({
        'Id': test_ids,
        'Recovery Index': test_predictions
    })
    submission_path = '/Users/garvrajput/StudioProjects/ML PROJ/linear_regression_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Final submission saved to '{submission_path}'")
    print(f"Test predictions range: {test_predictions.min():.4f} to {test_predictions.max():.4f}")
    print(f"Test predictions mean: {test_predictions.mean():.4f}")

    return {
        'model': best_model,
        'best_performance': best_performance,
        'best_method': best_method,
        'metrics': {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_mae': val_mae
        },
        'cv_results': {
            '5fold_cv': cv_scores_5fold,
            '10fold_cv': cv_scores_10fold,
        },
        'performance_comparison': {
            'validation_mse': val_performance,
            'cv_5fold_mse': cv_5fold_performance,
            'cv_10fold_mse': cv_10fold_performance,
            #'loocv_mse': loocv_mse
        },
        'submission_path': submission_path,
        'test_predictions': test_predictions
    }


def main():
    _ = train_and_evaluate_linear_regression()


if __name__ == "__main__":
    main()



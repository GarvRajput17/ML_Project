import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_preprocessing import DataPreprocessor


def run_elastic_net_cv(random_state: int = 42):
    print("Running Polynomial Regression with Elastic Net on Cleaned Data...")
    print("=" * 80)

    # Data - Using cleaned data
    prep = DataPreprocessor()
    train_df, test_df = prep.load_data(
        './data/train_cleaned.csv',
        './data/test_cleaned.csv'
    )
    prep.handle_categorical_variables()
    prep.create_feature_columns()
    prep.create_additional_features()
    X_train, X_val, y_train, y_val = prep.split_data(test_size=0.3)  # 70-30 split for linear models
    X_test_final = prep.prepare_test_data()

    # Combine train+val for final refit later
    X_full = np.vstack([X_train.values, X_val.values])
    y_full = np.concatenate([y_train.values, y_val.values])

    # Elastic Net Pipeline
    elastic_net_pipe = Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(random_state=random_state, max_iter=10000))
    ])

    # Search space for Elastic Net
    degrees = [1, 2, 3]
    alphas = np.logspace(-3, 3, 13)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # L1 ratio: 0=Ridge, 1=Lasso
    
    param_grid_elastic = {
        'poly__degree': degrees,
        'model__alpha': alphas,
        'model__l1_ratio': l1_ratios
    }

    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Grid search
    gs_elastic = GridSearchCV(
        elastic_net_pipe,
        param_grid_elastic,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=1
    )

    print("Fitting Elastic Net grid...")
    gs_elastic.fit(X_train, y_train)
    print(f"Best Elastic Net params: {gs_elastic.best_params_}")
    print(f"Best Elastic Net CV MSE: {-gs_elastic.best_score_:.4f}")

    # Get best model
    best_model = gs_elastic.best_estimator_
    best_params = gs_elastic.best_params_
    print(f"\nSelected best model: Elastic Net")

    # Evaluate on holdout validation
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print("\nValidation performance of Elastic Net model:")
    print(f"Val MSE: {val_mse:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val R²: {val_r2:.4f}")

    # Cross-Validation Analysis (5-fold and 10-fold only)
    print("\n" + "="*50)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*50)
    
    # 5-Fold CV
    kfold_5 = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores_5fold = cross_validate(best_model, X_train, y_train, 
                                   cv=kfold_5, scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
                                   return_train_score=True)
    
    print("5-Fold Cross-Validation Results:")
    print(f"CV MSE (mean ± std): {-cv_scores_5fold['test_neg_mean_squared_error'].mean():.4f} ± {cv_scores_5fold['test_neg_mean_squared_error'].std():.4f}")
    print(f"CV R² (mean ± std): {cv_scores_5fold['test_r2'].mean():.4f} ± {cv_scores_5fold['test_r2'].std():.4f}")
    print(f"CV MAE (mean ± std): {-cv_scores_5fold['test_neg_mean_absolute_error'].mean():.4f} ± {cv_scores_5fold['test_neg_mean_absolute_error'].std():.4f}")
    
    # 10-Fold CV
    kfold_10 = KFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_scores_10fold = cross_validate(best_model, X_train, y_train, 
                                    cv=kfold_10, scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
                                    return_train_score=True)
    
    print("\n10-Fold Cross-Validation Results:")
    print(f"CV MSE (mean ± std): {-cv_scores_10fold['test_neg_mean_squared_error'].mean():.4f} ± {cv_scores_10fold['test_neg_mean_squared_error'].std():.4f}")
    print(f"CV R² (mean ± std): {cv_scores_10fold['test_r2'].mean():.4f} ± {cv_scores_10fold['test_r2'].std():.4f}")
    print(f"CV MAE (mean ± std): {-cv_scores_10fold['test_neg_mean_absolute_error'].mean():.4f} ± {cv_scores_10fold['test_neg_mean_absolute_error'].std():.4f}")

    # Performance Summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    val_performance = val_mse
    cv_5fold_performance = -cv_scores_5fold['test_neg_mean_squared_error'].mean()
    cv_10fold_performance = -cv_scores_10fold['test_neg_mean_squared_error'].mean()
    
    print("Performance Comparison:")
    print(f"Validation MSE: {val_performance:.4f}")
    print(f"5-Fold CV MSE: {cv_5fold_performance:.4f}")
    print(f"10-Fold CV MSE: {cv_10fold_performance:.4f}")
    
    print(f"\nBest Elastic Net parameters: {best_params}")
    print(f"Best performance (MSE): {val_performance:.4f}")

    # Create Elastic Net visualization
    def extract_degree_curve(grid_search):
        results = pd.DataFrame(grid_search.cv_results_)
        curves = {}
        for d in degrees:
            mask = results['param_poly__degree'] == d
            best_row = results.loc[mask].sort_values('mean_test_score', ascending=False).iloc[0]
            curves[d] = -best_row['mean_test_score']
        return curves

    elastic_curve = extract_degree_curve(gs_elastic)
    plt.figure(figsize=(8, 5))
    plt.plot(list(elastic_curve.keys()), list(elastic_curve.values()), 'o-', label='Elastic Net', color='green')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Best CV MSE (per degree)')
    plt.title('Elastic Net CV Performance vs Polynomial Degree')
    plt.legend()
    plot_path = '/Users/garvrajput/StudioProjects/ML PROJ/elastic_net_cv_degree_curve.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Retrain on full (train+val) with best params and create submission
    final_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=best_params['poly__degree'], include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(
            alpha=best_params['model__alpha'], 
            l1_ratio=best_params['model__l1_ratio'],
            random_state=random_state, 
            max_iter=10000
        ))
    ])

    final_pipe.fit(X_full, y_full)
    test_pred = final_pipe.predict(X_test_final)
    submission = pd.DataFrame({'Id': test_df['Id'].values, 'Recovery Index': test_pred})
    submit_path = '/Users/garvrajput/StudioProjects/ML PROJ/elastic_net_submission.csv'
    submission.to_csv(submit_path, index=False)
    print(f"Final submission saved to '{submit_path}'")
    print(f"Test predictions range: {test_pred.min():.4f} to {test_pred.max():.4f}")
    print(f"Test predictions mean: {test_pred.mean():.4f}")

    return {
        'best_model_name': 'Elastic Net',
        'best_params': best_params,
        'best_model': best_model,
        'best_performance': val_performance,
        'val_metrics': {'mse': val_mse, 'mae': val_mae, 'r2': val_r2},
        'cv_results': {
            '5fold_cv': cv_scores_5fold,
            '10fold_cv': cv_scores_10fold
        },
        'performance_comparison': {
            'validation_mse': val_performance,
            'cv_5fold_mse': cv_5fold_performance,
            'cv_10fold_mse': cv_10fold_performance
        },
        'degree_curve_plot': plot_path,
        'submission_path': submit_path,
        'test_predictions': test_pred
    }


def main():
    _ = run_elastic_net_cv()


if __name__ == "__main__":
    main()

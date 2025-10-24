# 🏥 Patient Recovery Prediction - ML Project

A machine learning project for predicting patient recovery indices using multiple regression models with automated hyperparameter tuning and feature engineering.

---

## 📋 Overview

This project implements **8 machine learning models** to predict patient recovery based on therapy hours, health scores, lifestyle activities, sleep patterns, and follow-up sessions. Each model includes automated hyperparameter tuning, 5-fold cross-validation, and comprehensive visualizations.

---

## 📁 Project Structure

```
ML_Project/
│
├── data/
│   ├── train.csv                    # Training dataset (8000 samples)
│   ├── test.csv                     # Test dataset (2000 samples)
│   └── data_preprocessing.py        # Data preprocessing pipeline
│
├── linear_regression/               # Linear Regression model
├── ridge/                           # Ridge Regression model
├── lasso/                           # Lasso Regression model
├── decision_tree/                   # Decision Tree model
├── random_forest/                   # Random Forest model
├── xg_boost/                        # XGBoost model
├── adaboost/                        # AdaBoost model
│
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/GarvRajput17/ML_Project.git
cd ML_Project
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

**Important**: Always run from the **main directory** (`ML_Project/`) using the `-m` flag.

```bash
# Linear Regression
python -m linear_regression.linear_regression_model

# Ridge Regression
python -m ridge.ridge_model

# Lasso Regression
python -m lasso.lasso_model

# Decision Tree
python -m decision_tree.decision_tree_model

# Random Forest
python -m random_forest.random_forest_model

# XGBoost
python -m xg_boost.xgboost_model

# AdaBoost
python -m adaboost.adaboost_model
```

### Run All Models
```bash
# Run all models sequentially
python -m linear_regression.linear_regression_model && \
python -m ridge.ridge_model && \
python -m lasso.lasso_model && \
python -m decision_tree.decision_tree_model && \
python -m random_forest.random_forest_model && \
python -m xg_boost.xgboost_model && \
python -m adaboost.adaboost_model
```

---

## 📊 Dataset

### Features (5 Original)
1. **Therapy Hours** - Number of therapy hours received
2. **Initial Health Score** - Patient's baseline health score
3. **Lifestyle Activities** - Activity level (Active/Moderate/Sedentary)
4. **Average Sleep Hours** - Average hours of sleep per night
5. **Follow-Up Sessions** - Number of follow-up appointments

### Engineered Features (4 Additional)
Created automatically by the preprocessing pipeline:
1. **Therapy_Health_Interaction** - Therapy Hours × Initial Health Score
2. **Sleep_FollowUp_Interaction** - Sleep Hours × Follow-Up Sessions
3. **Health_Sleep_Ratio** - Health Score / (Sleep Hours + 1)
4. **Total_Engagement** - Therapy Hours + Follow-Up Sessions

**Total Features Used**: 9

### Target Variable
- **Recovery Index** - Continuous value (0-100) indicating patient recovery level

### Data Split
- **Training**: 6400 samples (80%)
- **Validation**: 1600 samples (20%)
- **Test**: 2000 samples (for prediction)

---

## 🤖 Models

| Model | Description | Training Time |
|-------|-------------|---------------|
| **Linear Regression** | Simple baseline model | < 1 min |
| **Ridge Regression** | L2 regularization, handles multicollinearity | 1-2 min |
| **Lasso Regression** | L1 regularization, automatic feature selection | 1-2 min |
| **Decision Tree** | Interpretable tree-based model | 3-5 min |
| **Random Forest** | Ensemble of decision trees | 5-10 min |
| **XGBoost** | Gradient boosting, typically best performance | 5-15 min |
| **AdaBoost** | Adaptive boosting ensemble | 3-5 min |

### What Each Model Does
- Loads and preprocesses data using centralized pipeline
- Trains basic model with default parameters
- Performs automated hyperparameter tuning (GridSearchCV)
- Runs 5-fold cross-validation
- Generates comprehensive visualizations (6-panel analysis)
- Creates Kaggle-ready submission file

---

## 📦 Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
xgboost>=1.7.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📈 Model Outputs

Each model creates 3 files in its directory:
1. **Coefficients/Feature Importance Plot** - Shows which features matter most
2. **6-Panel Analysis** - Comprehensive performance visualization
3. **Submission CSV** - Kaggle-ready predictions (unrounded)

---

## 🔧 Troubleshooting

### `ModuleNotFoundError`
**Solution**: Make sure you're in the main `ML_Project/` directory and using `-m` flag.

### `FileNotFoundError: train.csv`
**Solution**: Ensure `data/train.csv` and `data/test.csv` exist in the data folder.

### `ImportError: DataPreprocessor`
**Solution**: Check that `data/data_preprocessing.py` exists with the `DataPreprocessor` class.

---

## 🎯 Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/GarvRajput17/ML_Project.git
cd ML_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a model
python -m linear_regression.linear_regression_model

# 4. Check output in the model's directory
```

---

## 👥 Contributors

- **Garv Rajput** - [GarvRajput17](https://github.com/GarvRajput17)
- **Nirbhay Sharma** - [NirbhaySharma504](https://github.com/NirbhaySharma504)

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 📞 Support

- **Issues**: [Report here](https://github.com/GarvRajput17/ML_Project/issues)
- **Questions**: Open a GitHub issue

---

**Happy Modeling! 🚀**

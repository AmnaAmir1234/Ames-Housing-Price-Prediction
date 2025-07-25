# Ames Housing Price Prediction: Complete ML Pipeline

A comprehensive machine learning pipeline for predicting house prices using the Ames Housing dataset. This project implements the full data science workflow from data preprocessing to model deployment, featuring advanced techniques including feature selection, model comparison, explainable AI, and ensemble methods.

## Repository Information

**Repository Name**: `ames-housing-price-prediction-ml-pipeline`

**Description**: Complete machine learning pipeline for house price prediction using Ames Housing dataset with advanced feature selection, model comparison, XAI analysis, and ensemble methods

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Model Performance](#model-performance)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements a complete machine learning pipeline for house price prediction using the Ames Housing dataset. The pipeline includes data preprocessing, comprehensive feature selection, multiple model training and comparison, explainable AI analysis, and ensemble model development.

### Key Highlights

- **Complete ML Pipeline**: From raw data to production-ready models
- **Advanced Feature Selection**: Multi-phase feature selection using statistical tests and LASSO regularization
- **Model Comparison**: Comparison of 8 different algorithms including interpretable and black-box models
- **Explainable AI**: SHAP, LIME, and Partial Dependence Plot analysis for model interpretability
- **Ensemble Methods**: Voting regressor combining best-performing models
- **Comprehensive Evaluation**: Multiple metrics and visualization techniques

## Features

### Data Preprocessing
- Missing value analysis and imputation
- Outlier detection using IQR method
- One-hot encoding for categorical variables
- Feature normalization using MinMax scaling

### Feature Selection
- **Phase 1**: Variance threshold filtering and correlation analysis
- **Phase 2**: Statistical tests (F-regression, Mutual Information)
- **Phase 3**: LASSO regularization with cross-validation
- Comprehensive feature importance analysis

### Model Development
- **Interpretable Models**: Linear Regression, Ridge Regression, Decision Tree
- **Black-box Models**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Cross-validation and overfitting analysis
- Performance comparison with multiple metrics

### Explainable AI (XAI)
- **SHAP Analysis**: Global and local feature importance
- **LIME**: Local interpretable model-agnostic explanations
- **Partial Dependence Plots**: Feature effect visualization
- **Feature Interaction Analysis**: 2D interaction plots

### Ensemble Methods
- Voting regressor with optimized weights
- Performance comparison between individual and ensemble models

## Dataset

The project uses the **Ames Housing Dataset**, which contains house sale prices for residential properties in Ames, Iowa from 2006 to 2010.

- **Target Variable**: SalePrice
- **Features**: 80+ features including house characteristics, location, and quality ratings
- **Size**: 2,930 observations

## Project Structure

```
ames-housing-price-prediction/
├── data/
│   └── AmesHousing.csv
├── notebooks/
│   └── complete_pipeline.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── xai_analysis.py
│   └── ensemble_methods.py
├── results/
│   ├── model_performance_results.csv
│   ├── selected_features.csv
│   └── visualizations/
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install shap lime eli5 pdpbox
```

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/AmnaAmir1234/ames-housing-price-prediction-ml-pipeline.git
cd ames-housing-price-prediction-ml-pipeline
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Ames Housing dataset and place it in the `data/` directory.

## Usage

### Running the Complete Pipeline

Execute the complete pipeline by running the main script or notebook:

```python
# Run individual components
python src/data_preprocessing.py
python src/feature_selection.py
python src/model_training.py
python src/xai_analysis.py
python src/ensemble_methods.py
```

### Key Functions

#### Data Preprocessing
```python
# Load and preprocess data
df = pd.read_csv('AmesHousing.csv')
df_processed = preprocess_data(df)
```

#### Feature Selection
```python
# Multi-phase feature selection
selected_features = feature_selection_pipeline(df_processed, target='SalePrice')
```

#### Model Training
```python
# Train and compare multiple models
results = train_and_compare_models(X_train, X_test, y_train, y_test)
```

#### XAI Analysis
```python
# Explainable AI analysis
xai_results = run_xai_analysis(model, X_test, y_test, feature_names)
```

## Pipeline Components

### 1. Data Preprocessing
- **Missing Value Analysis**: Identifies and visualizes missing data patterns
- **Missing Value Imputation**: Uses mean/median for numerical and mode for categorical features
- **Outlier Handling**: IQR-based outlier detection for robust imputation
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: MinMax normalization for numerical features

### 2. Feature Selection
- **Variance Threshold**: Removes low-variance features
- **Correlation Analysis**: Eliminates highly correlated features (>0.95)
- **Statistical Tests**: F-regression and Mutual Information scoring
- **LASSO Regularization**: L1 regularization with cross-validation
- **Final Selection**: Reduces features from 80+ to optimal subset

### 3. Model Training and Comparison
- **Interpretable Models**: Linear Regression, Ridge Regression, Decision Tree
- **Black-box Models**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Evaluation Metrics**: R², RMSE, Cross-validation scores
- **Overfitting Analysis**: Train vs. test performance comparison

### 4. Explainable AI Analysis
- **SHAP Values**: Global and local feature importance with interaction effects
- **LIME Explanations**: Local interpretable explanations for individual predictions
- **Partial Dependence Plots**: Visualization of feature effects on predictions
- **Feature Interactions**: 2D interaction analysis between important features

### 5. Ensemble Methods
- **Voting Regressor**: Combines Linear Regression, Ridge Regression, and CatBoost
- **Weighted Averaging**: Optimized weights based on individual model performance
- **Performance Evaluation**: Comparison with individual models

## Model Performance

### Individual Model Results
| Model | Test R² | Test RMSE | Category |
|-------|---------|-----------|----------|
| CatBoost | 0.8924 | 18,247 | Black-box |
| Random Forest | 0.8891 | 18,502 | Black-box |
| Ridge Regression | 0.8654 | 20,387 | Interpretable |
| Linear Regression | 0.8651 | 20,401 | Interpretable |

### Ensemble Model Performance
- **Ensemble R²**: 0.8945
- **Ensemble RMSE**: 18,012
- **Improvement**: 2.1% over best individual model

## Results

### Key Findings

1. **Feature Selection**: Reduced features from 80+ to 30 most important features
2. **Best Individual Model**: CatBoost with R² = 0.8924
3. **Best Ensemble**: Voting Regressor with R² = 0.8945
4. **Most Important Features**: Overall Quality, Ground Living Area, Garage Cars
5. **Model Interpretability**: Linear models provide clear coefficient interpretation

### Generated Outputs

- **Feature Selection Results**: `selected_features.csv`
- **Model Performance**: `model_performance_results.csv`
- **Feature Coefficients**: `interpretable_models_coefficients.xlsx`
- **Visualizations**: Comprehensive plots for analysis and presentation

## Key Technologies

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: Matplotlib, Seaborn
- **Explainable AI**: SHAP, LIME, ELI5
- **Model Selection**: Cross-validation, Grid Search

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ames Housing Dataset from Iowa State University
- Scikit-learn community for machine learning tools
- SHAP and LIME libraries for explainable AI
- Open source ML community for continuous innovation

## Contact

For questions or collaboration opportunities, please reach out through GitHub issues or pull requests.

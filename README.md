# AutoML Pro - Automated Machine Learning for Classification

<div align="center">

![AutoML Pro](https://img.shields.io/badge/AutoML-Pro-6366f1?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**End-to-end automated machine learning pipeline for supervised classification tasks**

[Live Demo](https://automlbyali.streamlit.app/) · [Report Bug](https://github.com/AliSafdarSaeed/AutoML-System/issues) · [Request Feature](https://github.com/AliSafdarSaeed/AutoML-System/issues)

</div>

---

## Overview

AutoML Pro is a production-grade automated machine learning system designed for classification tasks. Built with Streamlit, this application provides an intuitive interface for data scientists, analysts, and ML practitioners to build, evaluate, and compare machine learning models without writing code.

The system automates the complete ML pipeline—from data ingestion and exploratory analysis to model training and report generation—while maintaining full transparency and user control over preprocessing decisions.

---

## Features

### Data Ingestion & Validation
- CSV file upload with drag-and-drop interface
- Automatic data type detection and validation
- Real-time data health dashboard with key metrics
- Memory usage optimization for large datasets

### Automated Exploratory Data Analysis
- **Missing Value Analysis**: Per-feature and global percentage breakdown
- **Outlier Detection**: IQR-based statistical outlier identification
- **Correlation Matrix**: Interactive heatmap visualization
- **Distribution Plots**: Histograms for numerical features
- **Categorical Analysis**: Bar charts for categorical feature distributions
- **Class Distribution**: Target variable balance assessment

### Data Quality Detection & Resolution
Automated detection of data quality issues with configurable resolution strategies:

| Issue Type | Detection Method | Available Fixes |
|------------|------------------|-----------------|
| Missing Values | Null count per column | Mean, Median, Mode, Drop |
| Outliers | IQR method (1.5×IQR) | Remove, Cap, No action |
| Class Imbalance | Majority ratio threshold | Warning + recommendations |
| High Cardinality | Unique value ratio | Encoding suggestions |
| Constant Features | Variance analysis | Removal recommendations |

### Preprocessing Pipeline
- **Imputation**: Mean, median, mode, or constant value strategies
- **Outlier Handling**: Removal, capping at IQR bounds, or retention
- **Feature Scaling**: StandardScaler or MinMaxScaler
- **Categorical Encoding**: One-Hot Encoding or Ordinal Encoding
- **Train-Test Split**: Configurable ratio (default 80/20) with stratification

### Model Training & Hyperparameter Optimization

Seven classification algorithms with automated hyperparameter tuning:

| Algorithm | Hyperparameter Grid |
|-----------|---------------------|
| Logistic Regression | C: [0.1, 1, 10], max_iter: 1000 |
| K-Nearest Neighbors | n_neighbors: [3, 5, 7], weights: [uniform, distance] |
| Decision Tree | max_depth: [3, 5, 10, None], min_samples_split: [2, 5] |
| Naive Bayes | var_smoothing: [1e-9, 1e-8, 1e-7] |
| Random Forest | n_estimators: [50, 100], max_depth: [5, 10, None] |
| Support Vector Machine | C: [0.1, 1], kernel: [rbf, linear] |
| Rule-Based Baseline | strategy: [most_frequent, stratified] |

**Optimization Methods:**
- GridSearchCV with cross-validation
- 3-fold stratified cross-validation
- F1-weighted scoring for multi-class problems

### Model Evaluation & Comparison
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Per-model visualization
- **ROC Curves**: Binary classification performance curves
- **Training Time**: Computational cost comparison
- **Leaderboard**: Ranked model comparison table
- **Exportable Results**: CSV download of all metrics

### Report Generation
Comprehensive PDF report containing:
- Dataset overview and statistics
- EDA findings and visualizations
- Detected issues and applied fixes
- Preprocessing configuration details
- Model comparison tables with metrics
- Best model recommendation with justification

---

## System Architecture

```
AutoML-System/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── caching.py                 # Performance optimization layer
├── assets/
│   └── themes/                # CSS theme files (dark/light)
├── data_utils/
│   ├── analysis.py            # Data profiling functions
│   ├── preprocessing.py       # Data transformation pipeline
│   ├── visualizations.py      # Plotly chart generators
│   └── reporting.py           # PDF report generation
├── models/
│   ├── model_configs.py       # Algorithm configurations
│   ├── trainer.py             # Training orchestration
│   └── visualizations.py      # Model performance charts
└── modules/
    ├── components.py          # Reusable UI components
    ├── ingestion_ui.py        # Data upload interface
    ├── eda_ui.py              # Exploratory analysis interface
    ├── quality_ui.py          # Data quality gate interface
    ├── training_ui.py         # Model training interface
    ├── reporting_ui.py        # Report generation interface
    └── recommendations.py     # AI-powered suggestions
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AliSafdarSaeed/AutoML-System.git
   cd AutoML-System
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   streamlit run main.py
   ```

5. **Access the application**
   
   Open your browser and navigate to `http://localhost:8501`

---

## Usage Guide

### Step 1: Data Upload
1. Navigate to the **Upload** section
2. Drag and drop your CSV file or click to browse
3. Review the data health dashboard for initial insights

### Step 2: Exploratory Analysis
1. Select your target variable for classification
2. Review correlation matrix and distribution plots
3. Examine the data quality summary

### Step 3: Data Quality Gate
1. Review detected issues (missing values, outliers, etc.)
2. Configure preprocessing options for each issue
3. Apply recommended fixes or customize settings
4. Approve configuration to generate clean dataset

### Step 4: Model Training
1. Select algorithms to train (AI recommendations provided)
2. Enable/disable GridSearchCV optimization
3. Click "Train Models" to start training
4. Review the real-time leaderboard

### Step 5: Report Generation
1. Review the project summary
2. Download the comprehensive PDF report

---

## Sample Datasets

For testing, we recommend the following datasets:

| Dataset | Source | Size | Classes |
|---------|--------|------|---------|
| Weather Dataset | [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) | 145K rows | 2 |
| Iris Dataset | [UCI](https://archive.ics.uci.edu/ml/datasets/iris) | 150 rows | 3 |
| Titanic Dataset | [Kaggle](https://www.kaggle.com/c/titanic) | 891 rows | 2 |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend Framework | Streamlit 1.28+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn 1.3+ |
| Visualization | Plotly, Seaborn, Matplotlib |
| Report Generation | FPDF |
| Styling | Custom CSS (Dark/Light themes) |

---

## Performance Considerations

- **Large Datasets**: Outlier detection uses sampling (50K rows) for datasets exceeding threshold
- **Caching**: Expensive computations are cached using `@st.cache_data`
- **Memory**: Optimized DataFrame operations for reduced memory footprint
- **Parallel Processing**: GridSearchCV uses `n_jobs=-1` for multi-core utilization

---

## Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Navigate to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Select `main.py` as the entry point
5. Deploy

The application is currently deployed at: **[https://automl-system.streamlit.app/](https://automl-system.streamlit.app/)**

---



## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

This project is developed as part of the CS-245 Machine Learning course (Fall 2025).

---

## Authors

- **Ali Safdar Saeed   [https://github.com/AliSafdarSaeed]** 
- **Saqib Mahmood      [https://github.com/saqibm-bh]**


---



# MIMIC-III Sepsis Analysis

A machine learning project for analyzing and predicting sepsis outcomes using the MIMIC-III Clinical Database. This project implements multiple ML algorithms including XGBoost, CatBoost, and ensemble methods to predict sepsis-related outcomes in ICU patients.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)

## 🔍 Overview

This project analyzes sepsis patients from the MIMIC-III database, focusing on:
- Extracting sepsis patients based on ICD-9 codes (99591, 99592, 78552)
- Feature engineering from clinical data
- Building predictive models using various machine learning algorithms
- Handling class imbalance using SMOTE
- Model evaluation and comparison

## 📊 Dataset

### MIMIC-III Clinical Database

The project uses the [MIMIC-III Clinical Database](https://mimic.physionet.org/) v1.4, a freely accessible critical care database containing de-identified health data from ~60,000 ICU stays.

**Sepsis Classification:**
- **99591**: Sepsis
- **99592**: Severe Sepsis  
- **78552**: Septic Shock

### Data Files Used

- `patients.csv` - Patient demographic information
- `diagnoses_icd.csv` - ICD-9 diagnosis codes
- `admissions.csv` - Hospital admission details
- Additional clinical measurements and lab results

### Extracted Dataset

- **Total sepsis patients**: 7,770 records
- **Features**: 106 clinical variables including:
  - Vital signs (heart rate, blood pressure, temperature)
  - Laboratory values (glucose, lactate, creatinine)
  - SOFA scores
  - Demographic information
  - Treatment indicators (RRT, fluid bolus)

## ✨ Features

- **Data Extraction**: Automated extraction of sepsis patients from MIMIC-III
- **Data Preprocessing**: 
  - Handling missing values
  - ICD-9 code validation and cleaning
  - Age filtering (adults only, ≥18 years)
  - Feature normalization
- **Machine Learning Models**:
  - XGBoost Classifier
  - CatBoost Classifier
  - Neural Networks (TensorFlow/Keras)
  - Ensemble methods (Stacking)
- **Class Imbalance Handling**: SMOTE oversampling
- **Model Evaluation**: 
  - Accuracy, ROC-AUC scores
  - Confusion matrices
  - Classification reports
  - Cross-validation

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Access to MIMIC-III database (requires credentialing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mimic-analysis.git
cd mimic-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download MIMIC-III data:
   - Request access at [PhysioNet](https://physionet.org/content/mimiciii/)
   - Download the clinical database files
   - Update the `path` variable in the notebooks to point to your MIMIC-III directory

## 💻 Usage

### 1. Extract Sepsis Patients

Run the extraction notebook to identify and extract sepsis patients:

```bash
jupyter notebook ExtractSepsisPatients.ipynb
```

This will generate `sepsis_patients.csv` containing filtered patient data.

### 2. Data Loading and Exploration

```bash
jupyter notebook LoadingCSVs.ipynb
```

### 3. Model Training

#### XGBoost Model
```bash
jupyter notebook XGBoost.ipynb
```

#### CatBoost Model
```bash
jupyter notebook CatBoost.ipynb
```

#### Stacking Ensemble
```bash
jupyter notebook StackingTest.ipynb
```

### 4. Main Analysis

```bash
jupyter notebook mimic.ipynb
```

## 📁 Project Structure

```
mimic-analysis/
│
├── ExtractSepsisPatients.ipynb    # Patient extraction and filtering
├── LoadingCSVs.ipynb              # Data loading utilities
├── mimic.ipynb                    # Main analysis notebook
├── XGBoost.ipynb                  # XGBoost model implementation
├── XGBtest.ipynb                  # XGBoost testing and tuning
├── CatBoost.ipynb                 # CatBoost model implementation
├── StackingTest.ipynb             # Ensemble stacking methods
├── beforeSofa.ipynb               # Pre-SOFA score analysis
│
├── extractedMimic.csv             # Processed MIMIC data
├── sepsis_patients.csv            # Extracted sepsis patient cohort
├── logs.log                       # Application logs
│
├── catboost_info/                 # CatBoost training artifacts
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🤖 Models

### XGBoost
- Gradient boosting implementation
- Hyperparameter tuning via GridSearchCV
- Feature importance analysis

### CatBoost
- Categorical feature handling
- Built-in cross-validation
- Training visualization

### Neural Networks
- TensorFlow/Keras implementation
- Dense layer architecture
- Binary classification output

### Ensemble Methods
- Stacking classifier
- Multiple base estimators
- Meta-learner optimization

## 📈 Results

The models are evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives
- **F1 Score**: Harmonic mean of precision and recall
- **Cross-Validation**: Stratified K-fold validation

*Note: Specific performance metrics can be found in individual notebook outputs.*

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
catboost
tensorflow
keras
imbalanced-learn
numba
```

Create a `requirements.txt` file with specific versions:
```bash
pip freeze > requirements.txt
```

## ⚠️ Important Notes

- **Data Privacy**: MIMIC-III contains de-identified patient data. Ensure compliance with data use agreements.
- **Credentialing**: Access to MIMIC-III requires completion of the CITI "Data or Specimens Only Research" course.
- **Computational Resources**: Some models (especially ensemble methods) may require significant computational resources.

## 📄 License

This project is for educational and research purposes. The MIMIC-III database has its own data use agreement that must be followed.

## 🙏 Acknowledgments

- **MIMIC-III Database**: Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet.
- The MIT Laboratory for Computational Physiology
- PhysioNet

## 📚 References

1. Johnson, A. E. W., et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.
2. Seymour, C. W., et al. (2016). Assessment of Clinical Criteria for Sepsis. JAMA, 315(8), 762-774.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes only and should not be used for clinical decision-making.

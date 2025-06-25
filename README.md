# Credit Card Fraud Detection

This project focuses on detecting fraudulent transactions using machine learning models, specifically Decision Tree and XGBoost classifiers. The dataset used is the well-known `creditcard.csv` dataset, which contains credit card transactions made by European cardholders in September 2013.

---

## Dataset

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Features**: 30 anonymized numerical input features + 1 output label (`Class`)
- **Target**:
  - `0` = Valid transaction
  - `1` = Fraudulent transaction

---

## Project Workflow

### 1. Data Loading and Exploration
- Used `pandas` to load the dataset using `read_csv()`.
- Explored the dataset structure using `.info()`, `.shape`, `.describe()`.
- Checked for missing values using `.isnull().sum()`.

### 2. Data Visualization
- Histograms plotted for all features using `.hist()` to understand feature distributions.
- Correlation heatmap generated using `seaborn.heatmap()` to observe relationships among features.

### 3. Fraud Analysis
- Number of fraud and valid transactions identified via boolean indexing.
- Imbalanced nature of the dataset was observed: very few fraud cases relative to valid ones.

### 4. Data Preparation
- Separated dataset into:
  - `X`: Features (excluding `Class`)
  - `Y`: Target (`Class`)
- Split the data using `train_test_split()` (80% training, 20% testing).

### 5. Modeling and Evaluation

#### A. Decision Tree Classifier
- Trained using `DecisionTreeClassifier(max_depth=6)` for controlled complexity.
- Evaluated using:
  - Accuracy score
  - Confusion matrix
  - Classification report
- **Accuracy:** ~98.81%
- **F1-score for Fraud Class:** 84%
- **Precision for Fraud Class:** 90%
- **Recall for Fraud Class:** 80%

#### B. Standard Scaler + Decision Tree
- Features scaled using `StandardScaler` before fitting into the classifier.
- Slight performance improvement and better generalization.

#### C. XGBoost Classifier
- Can be trained if `xgboost` package is installed.
- Generally achieves better performance with imbalanced datasets.

### 6. Evaluation Metrics
- **Confusion Matrix**: Visualized using `seaborn.heatmap()`.
- **Classification Report**:
  - Shows precision, recall, and F1-score for each class.
  - Especially important for evaluating fraud detection performance.

---

## Observations

- The dataset is highly imbalanced, so standard accuracy is not a reliable metric alone.
- Decision Tree performed reasonably well, but may overfit if max depth is not controlled.
- Feature scaling helped improve stability.
- Fraud detection recall can be further improved with more advanced techniques like SMOTE, under-sampling, or ensemble methods.

---

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost (optional)

Install dependencies using:

```bash
pip install -r requirements.txt

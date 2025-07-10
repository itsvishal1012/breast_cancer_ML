
# 🧠 Breast Cancer Classification using Machine Learning

This project applies supervised machine learning to classify tumors as **benign** or **malignant** based on features extracted from digitized images of a breast mass. It explores multiple classification models and improves performance through hyperparameter tuning.

---

## 📁 Dataset

- **Source**: [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Size**: 569 samples, 32 columns
- **Target Variable**: `diagnosis` (`M` = Malignant, `B` = Benign)
- **Features**: 30 numerical values computed from digitized images of a fine needle aspirate of breast mass.

---

## 🎯 Project Goals

- Preprocess the dataset for machine learning
- Train multiple classification algorithms
- Evaluate their performance using common classification metrics
- Use GridSearchCV and RandomizedSearchCV to tune hyperparameters
- Identify the best-performing model

---

## 🧪 Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Gaussian Naive Bayes

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

---

## 🔄 Workflow

### 1. Data Preprocessing
- Dropped the `id` column as it's not predictive.
- Converted the `diagnosis` column to binary (Malignant = 1, Benign = 0).
- Standardized features using `StandardScaler`.

### 2. Model Training & Initial Evaluation

| Model                  | Accuracy   | Precision  | Recall    | F1 Score  |
|-------------------------|------------|------------|-----------|-----------|
| **Support Vector Machine** | **0.9825** | **1.0000** | 0.9535    | **0.9762** |
| Logistic Regression     | 0.9737     | 0.9762     | 0.9535    | 0.9647     |
| Random Forest           | 0.9649     | 0.9756     | 0.9302    | 0.9524     |
| Naive Bayes             | 0.9649     | 0.9756     | 0.9302    | 0.9524     |
| Decision Tree           | 0.9386     | 0.9091     | 0.9302    | 0.9195     |

### 3. Hyperparameter Tuning

#### GridSearchCV for Random Forest
- `n_estimators`: [100, 200]
- `max_depth`: [None, 10]
- `min_samples_split`: [2, 5]
- `class_weight`: ['balanced']

#### RandomizedSearchCV for SVC
- `C`: [0.1, 1, 10]
- `kernel`: ['linear', 'rbf']
- `gamma`: ['scale', 'auto']
- `class_weight`: ['balanced']

Used `StratifiedKFold` for cross-validation and `f1` score as the evaluation metric.

### 4. Post-Tuning Evaluation

| Model               | Accuracy   | Precision  | Recall    | F1 Score  |
|---------------------|------------|------------|-----------|-----------|
| **Tuned SVC**       | **0.9737** | **0.9762** | 0.9535    | **0.9647** |
| Tuned Random Forest | 0.9649     | 0.9756     | 0.9302    | 0.9524     |

---

## 📈 Visualizations

- Confusion matrices for both tuned models
- Classification reports to evaluate per-class precision, recall, and F1-score

---

## 🗂 Project Structure

```
├── breast-cancer.csv
├── breast_cancer_classification.ipynb
├── README.md
```

---

## 🚀 Getting Started

### Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the Notebook

1. Open `breast_cancer_classification.ipynb` in Jupyter Notebook or any Python IDE.
2. Ensure `breast_cancer.csv` is in the same directory.
3. Run all cells to see model training, tuning, and evaluation.

---

## 📬 Contact

If you found this useful or want to collaborate, feel free to reach out via email or LinkedIn!


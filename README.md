# ⛏️ Seismic Bumps Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting **seismic bumps** using machine learning techniques.

The dataset contains seismic and mining-related measurements such as seismic activity, energy, pulse measurements, number of bumps, and hazard-related attributes. The project performs data preprocessing, categorical encoding, class balancing using **SMOTE**, feature selection, feature scaling, and machine learning classification.

The main objective is to build a machine learning model that can classify whether a seismic event belongs to the target class.

---

## 🎯 Objectives

* Analyze the Seismic Bumps dataset.
* Perform data cleaning and preprocessing.
* Handle categorical variables using **One-Hot Encoding**.
* Remove duplicate records.
* Handle class imbalance using **SMOTE**.
* Select the most relevant features using **SelectKBest**.
* Scale selected features using **StandardScaler**.
* Train and compare multiple machine learning algorithms.
* Identify the best-performing classification model.

---

## 📊 Dataset

The project uses the **Seismic Bumps dataset** stored as:

```text
seismic-bumps.csv
```

### Dataset Size

* **Original records:** 2,584
* **Original features:** 19
* **Target variable:** `class`

### Features

The dataset contains the following attributes:

```text
seismic
seismoacoustic
shift
genergy
gpuls
gdenergy
gdpuls
ghazard
nbumps
nbumps2
nbumps3
nbumps4
nbumps5
nbumps6
nbumps7
nbumps89
energy
maxenergy
class
```

The target variable is:

```text
class
```

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Remove Duplicates
   ↓
One-Hot Encoding
   ↓
Label Encoding of Target
   ↓
SMOTE
   ↓
Feature Selection
   ↓
Standard Scaling
   ↓
Train/Test Split
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Best Model Selection
```

---

## 🧹 Data Preprocessing

### 1. Data Loading

The dataset is loaded using Pandas:

```python
data = pd.read_csv("seismic-bumps.csv")
```

The dataset contains **2,584 rows and 19 columns**.

### 2. Missing Value Check

Missing values were checked using:

```python
df.isnull().sum()
```

The notebook shows **0 missing values** in the dataset.

### 3. Duplicate Removal

Duplicate records were identified:

```python
df.duplicated().sum()
```

There were **6 duplicate records**, which were removed using:

```python
df = df.drop_duplicates()
```

---

## 🔤 Categorical Encoding

The categorical columns are:

```text
seismic
seismoacoustic
shift
ghazard
```

These categorical features were converted into numerical features using **One-Hot Encoding**.

```python
OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
```

The target `class` was converted using `LabelEncoder`.

---

## ⚖️ Handling Class Imbalance

The dataset has an imbalanced target class.

To balance the classes, **SMOTE (Synthetic Minority Oversampling Technique)** was applied.

```python
smote = SMOTE()

x_smote, y_smote = smote.fit_resample(
    df1.drop('class', axis=1),
    df1['class']
)
```

After SMOTE:

```text
Class 0 : 2414
Class 1 : 2414
```

Total balanced samples:

```text
4828
```

SMOTE helps the models learn the minority class more effectively by generating synthetic samples.

---

## 🔍 Feature Selection

The project uses **SelectKBest with ANOVA F-test (`f_classif`)** to select the 10 most important features.

```python
skb = SelectKBest(score_func=f_classif, k=10)
```

### Selected Features

The final 10 selected features are:

```text
genergy
gpuls
nbumps
nbumps2
nbumps3
energy
seismic_a
seismic_b
shift_N
shift_W
```

These features were selected based on their statistical relationship with the target class.

---

## 📏 Feature Scaling

The selected features were standardized using **StandardScaler**.

```python
ss = StandardScaler()

x_scaled = ss.fit_transform(x_selected)
```

Scaling transforms the features so that they have comparable ranges, which is useful for several machine learning algorithms.

---

## 🤖 Machine Learning Models

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)
7. Naive Bayes

---

## 📈 Model Performance

The notebook reports the following accuracy results for the model comparison:

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression | **69.67%** |
| Decision Tree       | **91.82%** |
| Random Forest       | **94.82%** |
| Gradient Boosting   | **90.27%** |
| KNN                 | **81.78%** |
| SVM                 | **67.18%** |
| Naive Bayes         | **62.11%** |

### 🏆 Best Model

**Random Forest Classifier**

```text
Accuracy: 94.
```
# Seismic-bumps

# MLAgent Suite

## Introduction
The MLAgent Suite is a collection of sophisticated machine learning agents designed to address a wide spectrum of predictive modeling tasks across diverse domains, including healthcare, finance, social sciences, and engineering. These agents — `MLAgentClassifier`, `MLAgentRegressor`, `MLAgentCountRegressor`, `MLAgentLongitudinalClassifier`, `MLAgentLongitudinalRegressor`, `MLAgentLongitudinalCountRegressor`, and `MLAgentSurvival` — are engineered to handle complex data structures, automate analytical workflows, and deliver robust, interpretable results. Each agent is tailored to specific problem types, from standard classification and regression to specialized longitudinal and survival analyses, ensuring flexibility and precision. This executive summary introduces the functionalities, strengths, and applications of these agents, highlighting their role as comprehensive tools for researchers, data scientists, and practitioners seeking to tackle advanced predictive challenges.

## Overview of the MLAgent Suite
The **MLAgent Suite** integrates state-of-the-art statistical, machine learning, and deep learning models within a unified framework, emphasizing automation, scalability, and interpretability. Below is a brief overview of each agent, outlining their primary purposes, key features, and target use cases.

---

### 1. MLAgentClassifier

**Purpose:** Designed for standard classification tasks, predicting categorical outcomes from cross-sectional data.

**Key Features:**
- **Model Support:** Logistic regression, random forests, support vector machines, neural networks.
- **Preprocessing:** Handles missing values (mean/mode imputation), categorical encoding (LabelEncoder), and feature scaling (StandardScaler).
- **Evaluation:** Computes accuracy, precision, recall, F1-score, and ROC-AUC with bootstrapped confidence intervals.
- **Visualization:** Confusion matrices, ROC curves, feature importance plots.
- **Persistence:** Supports model saving and loading for deployment.

**Applications:** Customer segmentation, fraud detection, disease diagnosis.

---

### 2. MLAgentRegressor

**Purpose:** Addresses standard regression tasks, predicting continuous outcomes from cross-sectional data.

**Key Features:**
- **Model Support:** Linear regression, random forests, gradient boosting, neural networks.
- **Preprocessing:** Imputation, encoding, and scaling, similar to MLAgentClassifier.
- **Evaluation:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² with confidence intervals.
- **Visualization:** Residual plots, prediction vs. actual scatter plots, feature importance visualizations.
- **Persistence:** Model serialization for reuse.

**Applications:** Predicting house prices, stock values, patient recovery metrics.

---

### 3. MLAgentCountRegressor

**Purpose:** Tailored for regression tasks involving count data, modeling non-negative integer outcomes.

**Key Features:**
- **Model Support:** Poisson regression, random forests (adapted for counts), and neural networks for count data.
- **Preprocessing:** Validates non-negative targets, imputes missing values, encodes categoricals, scales features.
- **Evaluation:** Poisson deviance, MSE, MAE, and R².
- **Visualization:** Count distributions, prediction vs. actual plots, feature importance.
- **Persistence:** Supports model saving/loading.

**Applications:** Predicting event frequencies like hospital visits, website clicks, insurance claims.

---

### 4. MLAgentLongitudinalClassifier

**Purpose:** Designed for classification tasks with longitudinal data, predicting categorical outcomes from time-series observations.

**Key Features:**
- **Model Support:** Random-effects logistic regression, longitudinal random forests, Hidden Markov Models, GPBoost, deep learning models (TCN, LSTM, GRU, Transformer).
- **Preprocessing:** Variable-length sequences, imputes missing values, encodes categoricals, scales features.
- **Evaluation:** AUC-ROC, AUPRC, accuracy, precision, recall, F1-score, with optimal threshold selection via Youden’s J statistic.
- **Visualization:** Target distribution bar plots, ROC/PRC curves, feature importance plots.
- **Persistence:** Saves models as .pkl or .pth files.

**Applications:** Longitudinal studies like predicting disease progression, customer churn, behavior in panel data.

---

### 5. MLAgentLongitudinalRegressor

**Purpose:** Targets regression tasks with longitudinal data, predicting continuous outcomes from temporal sequences.

**Key Features:**
- **Model Support:** Random-effects linear regression, longitudinal random forests, HMM, GPBoost, and deep learning models (TCN, LSTM, GRU, Transformer).
- **Preprocessing:** Variable-length sequences, imputes missing values, encodes categoricals, scales features.
- **Evaluation:** MAE, MSE, RMSE, R², with bootstrapped confidence intervals.
- **Visualization:** Outcome distribution histograms, prediction vs. actual scatter plots, feature importance bar charts.
- **Persistence:** Model serialization.

**Applications:** Tracking biomarkers, economic indicators, or behavioral metrics over time.

---

### 6. MLAgentLongitudinalCountRegressor

**Purpose:** Specialized for regression tasks with longitudinal count data, modeling time-series count outcomes.

**Key Features:**
- **Model Support:** Random-effects Poisson regression, longitudinal random forests, HMM, GPBoost, deep learning models (TCN, LSTM, GRU, Transformer).
- **Preprocessing:** Non-negative targets, variable-length sequences, imputes missing values, scales features.
- **Evaluation:** Poisson deviance, MSE, MAE, and R².
- **Visualization:** Count distribution histograms, prediction vs. actual scatter plots, feature importance visualizations.
- **Persistence:** Saves models as .pkl or .pth files.

**Applications:** Modeling temporal count data, such as seizure frequencies, daily sales trends.

---

### 7. MLAgentSurvival

**Purpose:** Engineered for survival analysis, modeling time-to-event data with censoring.

**Key Features:**
- **Model Support:** Cox Proportional Hazards, Random Survival Forests, Fast Survival SVM, deep learning models (DeepSurv, Survival CNN).
- **Preprocessing:** StratifiedShuffleSplit, scales features, prepares structured time-to-event formats.
- **Evaluation:** C-Index, Integrated Brier Score (IBS), Time-Dependent AUC with model selection by C-Index.
- **Visualization:** Kaplan-Meier survival curves, feature importance plots.
- **Persistence:** Saves models using joblib or .pth files.

**Applications:** Predicting patient survival, equipment failure times, customer churn.

---

## Strengths and Innovations
The MLAgent Suite stands out for its integration of advanced methodologies within a user-friendly framework:
- **Comprehensive Model Support:** Combines statistical, machine learning, and deep learning models.
- **Automation:** Automates preprocessing, model training, evaluation, and visualization.
- **Robust Evaluation:** Employs domain-specific metrics with bootstrapped confidence intervals.
- **Interpretability:** Provides intuitive visualizations for enhanced decision-making.
- **Scalability:** Supports variable-length sequences and GPU-accelerated deep learning.
- **Flexibility:** Allows for model customization to fit specific research needs.

---

## Applications and Impact
The MLAgent Suite addresses a broad range of predictive tasks:
- **Healthcare:** Diagnosing diseases, predicting biomarkers, hospital visit modeling, survival estimation.
- **Finance:** Stock price forecasting, fraud detection, predicting transaction counts.
- **Social Sciences:** Analyzing behavioral trends and panel survey outcomes.
- **Engineering:** Predicting equipment failures and maintenance event counts.

---

## Conclusion
The MLAgent Suite represents a significant advancement in predictive modeling, offering a cohesive set of tools that balance sophistication with accessibility. From standard classification and regression to complex longitudinal and survival analyses, the agents empower users to extract meaningful insights from diverse datasets. Their automated workflows, robust evaluations, and interpretable outputs make them indispensable for tackling real-world challenges. As data complexity grows, the MLAgent Suite is poised to evolve, integrating new models and techniques to remain at the forefront of machine learning innovation.

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   python check-and-install-packages.py

   # Or
   pip install -r requirements.txt
   ```

## Usage

Basic usage example: `Binary` and `Multiclass` classification task

```python
from mlagents import MLAgentClassifier
import pandas as pd

# Load your data
X = pd.DataFrame(your_features)
y = your_target

# Initialize MLAgent
agent = MLAgentClassifier()

# Load and preprocess data
agent.load_data(X, y, feature_names=your_feature_names)

# Train and evaluate models
agent.train_and_evaluate()

# Make predictions
predictions = agent.predict()
```

See the example files for more detailed usage:
- `example-MLAgent-iris.py`: Example using the Iris dataset
- `example-MLAgent-titanic.py`: Example using the Titanic dataset

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- scikit-learn
- tensorflow/keras
- pandas
- numpy
- matplotlib
- seaborn
- xgboost

## License

Copyright (c) [2025] [Jacques Muthusi, Daniel Maangi]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

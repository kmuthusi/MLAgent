# MLAgent

MLAgent is an automated machine learning (AutoML) tool that simplifies the process of training, evaluating, and comparing multiple machine learning models. It handles both classification and regression tasks with minimal user input, providing comprehensive insights and visualizations.

## Features

- **Multiple Model Support**:
  - Traditional ML models (Logistic Regression, Decision Trees, Random Forest, SVM, etc.)
  - Deep Learning models (MLP and CNN using Keras)
  - Ensemble methods (XGBoost, Gradient Boosting)

- **Automated Data Processing**:
  - Automatic handling of categorical and numerical features
  - Missing value imputation
  - Feature scaling
  - Label encoding for categorical variables

- **Comprehensive Evaluation**:
  - Accuracy metrics
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Feature importance analysis

- **Visualization Tools**:
  - Target distribution plots
  - Feature importance plots
  - Model performance visualizations

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   python check-and-install-packages.py

   # Or
   pip install -r requirements.txt
   ```

## Usage

Basic usage example:

```python
from MLAgent import MLAgent
import pandas as pd

# Load your data
X = pd.DataFrame(your_features)
y = your_target

# Initialize MLAgent
agent = MLAgent()

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

Copyright (c) [year] [copyright holder]

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

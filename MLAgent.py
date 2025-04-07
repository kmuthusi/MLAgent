import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.inspection import permutation_importance
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import Parallel, delayed
import warnings
import multiprocessing

# setup seed for reproducibity
np.random.seed(2024)

warnings.filterwarnings('ignore')

class MLAgent:
    def __init__(self, models_to_train=None):
        """Initialize with safe parallel processing defaults."""
        self.n_cores = multiprocessing.cpu_count()
        self.parallel_enabled = self.n_cores > 2
        self.n_jobs = max(1, self.n_cores - 1) if self.parallel_enabled else 1

        self.all_models = {
            'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'Sklearn ANN': MLPClassifier(max_iter=1000)
        }

        self.keras_model_types = ['Keras MLP', 'Keras CNN']
        self.models_to_train = models_to_train if models_to_train is not None else list(
            self.all_models.keys()) + self.keras_model_types
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_cat = None
        self.y_test_cat = None
        self.optimal_cutoff = 0.5
        self.feature_names = None
        self.label_encoders = {}
        self.num_imputer = None
        self.cat_imputer = None
        self.trained_models = {}
        self.target_encoder = None
        self.target_is_categorical = False
        self.num_classes = None

    def _train_model(self, name, model):
        """Helper function for training a single model."""
        try:
            if name in self.all_models:
                model.fit(self.X_train, self.y_train)
                pred = model.predict(self.X_test)
                score = accuracy_score(self.y_test, pred)
                return (name, model, score)
    
            elif name in self.keras_model_types:
                if name == 'Keras CNN':
                    # Reshape for Conv1D input
                    X_train_dl = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
                    X_test_dl = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                else:
                    # For MLP or other types that don't require reshaping
                    X_train_dl = self.X_train
                    X_test_dl = self.X_test
    
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(
                    X_train_dl,
                    self.y_train_cat if self.num_classes > 2 else self.y_train,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
    
                pred_proba = model.predict_proba(X_test_dl)
                if self.num_classes > 2:
                    pred = np.argmax(pred_proba, axis=1)
                else:
                    if pred_proba.ndim == 1:
                        pred_proba = pred_proba.reshape(-1, 1)
                    pred = (pred_proba[:, -1] >= 0.5).astype(int)
    
                score = accuracy_score(self.y_test, pred)
                return (name, model, score)
    
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            return (name, None, 0)

    def load_data(self, X, y, feature_names=None, test_size=0.3, random_state=42):
        """Prepare and split the data for training and testing."""
        print("Input X shape:", X.shape)
        print("Input y shape:", y.shape if hasattr(y, 'shape') else len(y))
        print("Input y type:", type(y))
    
        mask = ~pd.isna(y)
        if isinstance(mask, pd.Series):
            mask = mask.values
        X = X[mask]
        y = y[mask]
    
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray) and feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            else:
                raise ValueError("X must be a pandas DataFrame or numpy array with feature_names provided")
    
        self.feature_names = X.columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        self.target_is_categorical = not np.issubdtype(y.dtype, np.number)
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
        if self.numerical_columns:
            self.num_imputer = SimpleImputer(strategy='mean')
            self.X_train[self.numerical_columns] = self.num_imputer.fit_transform(self.X_train[self.numerical_columns])
            self.X_test[self.numerical_columns] = self.num_imputer.transform(self.X_test[self.numerical_columns])
    
        if self.categorical_columns:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            self.X_train[self.categorical_columns] = self.cat_imputer.fit_transform(
                self.X_train[self.categorical_columns])
            self.label_encoders = {}
            for col in self.categorical_columns:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.X_train[col] = oe.fit_transform(self.X_train[[col]])
                self.X_test[col] = oe.transform(self.X_test[[col]])
                self.label_encoders[col] = oe
    
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
    
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
    
        # Initialize and fit the target encoder
        self.target_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.y_train = self.target_encoder.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test = self.target_encoder.transform(self.y_test.reshape(-1, 1)).flatten()
    
        self.num_classes = len(np.unique(self.y_train))
        if self.num_classes > 2:
            self.y_train_cat = to_categorical(self.y_train, num_classes=self.num_classes)
            self.y_test_cat = to_categorical(self.y_test, num_classes=self.num_classes)
        else:
            self.y_train_cat = self.y_train
            self.y_test_cat = self.y_test
    
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
        print("Final X_train shape:", self.X_train.shape)
        print("Final y_train shape:", self.y_train.shape if hasattr(self.y_train, 'shape') else len(self.y_train))

    def display_target_distribution(self):
        """Display the distribution of the target outcome."""
        if self.target_is_categorical:
            y_train_display = self.target_encoder.inverse_transform(self.y_train.astype(int).reshape(-1, 1)).flatten()
        else:
            y_train_display = self.y_train
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_train_display)
        plt.title("Distribution of Target Outcome in Training Set")
        plt.xlabel("Target Class")
        plt.ylabel("Count")
        plt.show()

    def build_deep_learning_model(self, model_type):
        """Define and return a deep learning model based on type."""
        input_shape = (self.X_train.shape[1],) if model_type == 'Keras MLP' else (self.X_train.shape[1], 1)
        if model_type == 'Keras MLP':
            model = Sequential([
                Dense(64, activation='relu', input_shape=input_shape),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(self.num_classes if self.num_classes > 2 else 1,
                      activation='softmax' if self.num_classes > 2 else 'sigmoid')
            ])
        elif model_type == 'Keras CNN':
            model = Sequential([
                Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(self.num_classes if self.num_classes > 2 else 1,
                      activation='softmax' if self.num_classes > 2 else 'sigmoid')
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def find_optimal_cutoff(self, y_true, y_pred_proba):
        """Compute optimal probability cutoff maximizing sensitivity and specificity."""
        if self.num_classes > 2:
            print("Optimal cutoff computation not implemented for multi-class; using default 0.5.")
            return 0.5
        
        if y_pred_proba.shape[1] == 2:
            pos_prob = y_pred_proba[:, 1]
        elif y_pred_proba.shape[1] == 1:
            pos_prob = y_pred_proba[:, 0]
        else:
            raise ValueError("Unexpected shape for y_pred_proba")
        
        fpr, tpr, thresholds = roc_curve(y_true, pos_prob)
        youden_j = tpr + (1 - fpr) - 1
        optimal_idx = np.argmax(youden_j)
        return thresholds[optimal_idx]

    def plot_multiclass_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve for multiclass classification."""
        if self.num_classes <= 2:
            print("ROC curve not applicable for binary classification.")
            return
    
        # Compute ROC curve and ROC AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_pred_proba[:, i], pos_label=i)
            roc_auc[i] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
        # Plot ROC curves
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], label=f'{self.target_encoder.categories_[0][i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
    
    def plot_multiclass_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve for multiclass classification."""
        if self.num_classes <= 2:
            print("Precision-Recall curve not applicable for binary classification.")
            return
    
        # Binarize the output for multiclass Precision-Recall
        y_true_binarized = label_binarize(y_true, classes=np.arange(self.num_classes))
    
        # Compute Precision-Recall curve for each class
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            average_precision[i] = average_precision_score(y_true_binarized[:, i], y_pred_proba[:, i])
    
        # Plot Precision-Recall curves
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes):
            plt.plot(recall[i], precision[i], label=f'{self.target_encoder.categories_[0][i]} (AP = {average_precision[i]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multiclass Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()
    
    def train_and_evaluate(self):
        """Train models, using parallel for scikit-learn and sequential for Keras."""
        if self.y_train is None:
            raise ValueError("y_train is None; ensure load_data is called correctly")
    
        results = {}
        sklearn_models = [(name, self.all_models[name]) for name in self.models_to_train if name in self.all_models]
        keras_models = [(name, KerasClassifier(
            model=lambda: self.build_deep_learning_model(name),
            epochs=50,
            batch_size=32,
            verbose=0
        )) for name in self.models_to_train if name in self.keras_model_types]
    
        if sklearn_models:
            trained_sklearn = Parallel(n_jobs=self.n_jobs)(
                delayed(self._train_model)(name, model) for name, model in sklearn_models
            )
            for name, model, score in trained_sklearn:
                if model is not None:
                    self.trained_models[name] = model
                    results[name] = score
                    print(f"{name} Test Accuracy: {score:.4f}")
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = (name, model)
    
        for name, model in keras_models:
            trained_result = self._train_model(name, model)
            name, model, score = trained_result
            if model is not None:
                self.trained_models[name] = model
                results[name] = score
                print(f"{name} Test Accuracy: {score:.4f}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = (name, model)
    
        if self.best_model:
            best_name, best_model = self.best_model
            print(f"\nBest Model: {best_name} with Accuracy: {self.best_score:.4f}")
    
            if self.num_classes <= 2:  # Binary classification
                if best_name in self.all_models:
                    y_pred_proba = best_model.predict_proba(self.X_test)
                else:
                    X_test_dl = (self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                                 if best_name == 'Keras CNN' else self.X_test)
                    y_pred_proba = best_model.predict_proba(X_test_dl)
    
                # Output for binary classification
                self.optimal_cutoff = self.find_optimal_cutoff(self.y_test, y_pred_proba)
                print(f"Optimal Cutoff: {self.optimal_cutoff:.4f}")
                self.display_target_distribution()
                self.show_confusion_matrix(y_pred_proba)
                self.plot_roc_curve(y_pred_proba)
                self.plot_precision_recall_curve(y_pred_proba)
                self.compute_dams(y_pred_proba)
                self.show_feature_importance()
    
            else:  # Multiclass classification
                if best_name in self.all_models:
                    y_pred_proba = best_model.predict_proba(self.X_test)
                else:
                    X_test_dl = (self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                                 if best_name == 'Keras CNN' else self.X_test)
                    y_pred_proba = best_model.predict_proba(X_test_dl)
    
                # Output for multiclass classification
                print(f"Optimal Cutoff: {self.optimal_cutoff:.4f}")
                self.display_target_distribution()
                self.show_confusion_matrix(y_pred_proba)
                self.plot_multiclass_roc_curve(self.y_test, y_pred_proba)
                self.plot_multiclass_precision_recall_curve(self.y_test, y_pred_proba)
                self.compute_dams(y_pred_proba)  # Call to compute and print DAMs
                self.show_feature_importance()
    
        return results

    def show_confusion_matrix(self, y_pred_proba):
        """Display confusion matrix using optimal cutoff."""
        if self.num_classes > 2:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]) >= self.optimal_cutoff
            y_pred = y_pred.astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = self.target_encoder.categories_[0] if self.target_is_categorical else np.unique(self.y_train).astype(str)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix for {self.best_model[0]} (Cutoff: {self.optimal_cutoff:.4f})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def get_feature_importance(self, model, model_name):
        """Dynamically compute feature importance using permutation importance or native methods."""
        if hasattr(model, 'feature_importances_'):
            print(f"Using native feature_importances_ for {model_name}")
            return model.feature_importances_
        elif hasattr(model, 'coef_') and model.coef_.ndim == 1:
            print(f"Using native coef_ for {model_name}")
            return np.abs(model.coef_)
        
        print(f"Computing permutation importance for {model_name}...")
        X_test_perm = self.X_test  # Always 2D for permutation_importance
        perm_result = permutation_importance(model, X_test_perm, self.y_test, n_repeats=10, random_state=42, scoring='accuracy')
        return perm_result.importances_mean

    def show_feature_importance(self):
        """Plot top 10 feature importances for the best model."""
        if not self.best_model:
            print("No best model selected yet.")
            return
        best_name, best_model = self.best_model
        importance = self.get_feature_importance(best_model, best_name)
        feature_importance = pd.Series(importance, index=self.feature_names)
        top_10 = feature_importance.nlargest(10)
        plt.figure(figsize=(10, 6))
        top_10.plot(kind='barh')
        plt.title(f"Top 10 Feature Importances for {best_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()
        plt.show()

    def plot_roc_curve(self, y_pred_proba):
        """Plot ROC curve and compute AUC for binary classification."""
        if self.num_classes != 2:
            print("ROC curve not applicable for multi-class classification.")
            return
        pos_prob = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
        fpr, tpr, _ = roc_curve(self.y_test, pos_prob)
        auc = roc_auc_score(self.y_test, pos_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_precision_recall_curve(self, y_pred_proba):
        """Plot precision-recall curve and compute average precision for binary classification."""
        if self.num_classes != 2:
            print("Precision-Recall curve not applicable for multi-class classification.")
            return
        pos_prob = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
        precision, recall, _ = precision_recall_curve(self.y_test, pos_prob)
        ap = average_precision_score(self.y_test, pos_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()

    def compute_dams(self, y_pred_proba):
        """Compute diagnostic accuracy measures with 95% CI for binary and multiclass classification."""
        
        # For binary classification
        if self.num_classes == 2:
            # Use optimal cutoff to make predictions
            y_pred = (y_pred_proba[:, 1] >= self.optimal_cutoff).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            n = len(self.y_test)
            
            # Calculate metrics
            accuracy = (tp + tn) / n
            accuracy_ci = self.wilson_ci(tp + tn, n)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivity_ci = self.wilson_ci(tp, tp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_ci = self.wilson_ci(tn, tn + fp)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_ci = self.wilson_ci(tp, tp + fp)
            
            f1 = f1_score(self.y_test, y_pred)
    
            dams_df = pd.DataFrame({
                'Measure': ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score'],
                'Estimate': [accuracy, sensitivity, specificity, precision, f1],
                '95% CI': [f'({accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f})',
                           f'({sensitivity_ci[0]:.4f}, {sensitivity_ci[1]:.4f})',
                           f'({specificity_ci[0]:.4f}, {specificity_ci[1]:.4f})',
                           f'({precision_ci[0]:.4f}, {precision_ci[1]:.4f})',
                           'N/A']
            })
            
            print("\nDiagnostic Accuracy Measures (DAMS) for Binary Classification:")
            print(dams_df)
    
        # For multiclass classification
        else:
            # Get predicted classes
            y_pred = np.argmax(y_pred_proba, axis=1)
    
            # Calculate overall accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
    
            # Calculate precision, recall, and F1 score for each class
            precision = precision_score(self.y_test, y_pred, average=None)
            recall = recall_score(self.y_test, y_pred, average=None)
            f1 = f1_score(self.y_test, y_pred, average=None)
    
            # Calculate 95% CI for each class
            precision_ci = [self.wilson_ci(tp, tp + fp) for tp, fp in zip(
                [np.sum((y_pred == i) & (self.y_test == i)) for i in range(self.num_classes)],
                [np.sum((y_pred == i) & (self.y_test != i)) for i in range(self.num_classes)]
            )]
    
            recall_ci = [self.wilson_ci(tp, tp + fn) for tp, fn in zip(
                [np.sum((y_pred == i) & (self.y_test == i)) for i in range(self.num_classes)],
                [np.sum((y_pred != i) & (self.y_test == i)) for i in range(self.num_classes)]
            )]
    
            f1_ci = [self.wilson_ci(tp, tp + fp) for tp, fp in zip(
                [np.sum((y_pred == i) & (self.y_test == i)) for i in range(self.num_classes)],
                [np.sum((y_pred == i) & (self.y_test != i)) for i in range(self.num_classes)]
            )]
    
            # Create a DataFrame to hold the results
            dams_df = pd.DataFrame({
                'Class': [f'Class {i}' for i in range(self.num_classes)],
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Precision 95% CI': [f'({ci[0]:.4f}, {ci[1]:.4f})' for ci in precision_ci],
                'Recall 95% CI': [f'({ci[0]:.4f}, {ci[1]:.4f})' for ci in recall_ci],
                'F1 Score 95% CI': [f'({ci[0]:.4f}, {ci[1]:.4f})' for ci in f1_ci]
            })
    
            # Calculate macro averages
            dams_df.loc[len(dams_df)] = ['Macro Average', 
                                          dams_df['Precision'].mean(), 
                                          dams_df['Recall'].mean(), 
                                          dams_df['F1 Score'].mean(),
                                          'N/A', 'N/A', 'N/A']  # No CI for macro average
    
            print("\nDiagnostic Accuracy Measures (DAMS) for Multiclass Classification:")
            print(dams_df)

    @staticmethod
    def wilson_ci(pos, n, confidence=0.95):
        """Compute Wilson score interval for a proportion."""
        if n == 0:
            return (0, 0)
        z = norm.ppf(1 - (1 - confidence) / 2)
        phat = pos / n
        denominator = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denominator
        return (center - margin, center + margin)

    def predict(self, X_new=None):
        """Predict using the best model, returning labels in original format."""
        if not self.best_model:
            raise ValueError("No model has been trained yet.")
    
        best_name, best_model = self.best_model
    
        if X_new is None:
            X_new_scaled = self.X_test
            y_true = self.y_test
        else:
            if isinstance(X_new, pd.DataFrame):
                if not all(col in X_new.columns for col in self.feature_names):
                    raise ValueError("X_new must contain all columns from the training data")
                X_new = X_new[self.feature_names].copy()
                if self.numerical_columns:
                    X_new[self.numerical_columns] = self.num_imputer.transform(X_new[self.numerical_columns])
                if self.categorical_columns:
                    X_new[self.categorical_columns] = self.cat_imputer.transform(X_new[self.categorical_columns])
                    for col in self.label_encoders:
                        if col in X_new.columns:
                            X_new[col] = self.label_encoders[col].transform(X_new[[col]])
                X_new = X_new.values
            elif not isinstance(X_new, np.ndarray):
                raise ValueError("X_new must be a DataFrame or numpy array")
    
            # Scale new data
            X_new_scaled = self.scaler.transform(X_new)
            y_true = None
    
        # Prepare input for prediction based on model type
        if best_name == 'Keras CNN':
            # Reshape new data for prediction with Conv1D layer
            X_new_dl = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))
        else:
            # For other types (like MLP), no reshaping needed
            X_new_dl = X_new_scaled
    
        # Make predictions using the best model
        pred_proba = best_model.predict_proba(X_new_dl)
    
        # Determine predicted classes based on probabilities
        if self.num_classes > 2:
            pred = np.argmax(pred_proba, axis=1)
        else:
            if not hasattr(self, 'optimal_cutoff'):
                raise ValueError("Optimal cutoff not computed. Please call train_and_evaluate first.")
            pred = (pred_proba[:, -1] >= self.optimal_cutoff).astype(int)
    
        # Keep numerical predictions for accuracy computation
        pred_numerical = pred.copy()
    
        # Decode to categorical labels only for return/display, if applicable
        if self.target_is_categorical and self.target_encoder is not None:
            pred = self.target_encoder.inverse_transform(pred.reshape(-1, 1)).flatten()
    
        if y_true is not None:
            # Use numerical predictions for accuracy
            accuracy = accuracy_score(y_true, pred_numerical)
            print(f"Accuracy on test set with {best_name}: {accuracy:.4f}")
    
        return pred  # Return categorical predictions

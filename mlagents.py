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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, dump, load
import warnings
import multiprocessing
import os

np.random.seed(2024)
warnings.filterwarnings('ignore')

class KerasClassifierWrapper:
    """Wrapper to adjust KerasClassifier predict method for permutation importance."""
    def __init__(self, keras_classifier):
        self.model = keras_classifier

    def predict(self, X):
        pred_proba = self.model.predict_proba(X)
        return np.argmax(pred_proba, axis=1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

class MLAgentClassifier:
    def __init__(self, models_to_train=None, save_path="best_model"):
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
        self.optimal_cutoffs = None  # Changed to store per-class cutoffs
        self.feature_names = None
        self.label_encoders = {}
        self.num_imputer = None
        self.cat_imputer = None
        self.trained_models = {}
        self.target_encoder = None
        self.target_is_categorical = False
        self.num_classes = None
        self.save_path = save_path

    def _train_model(self, name, model):
        try:
            if name in self.all_models:
                model.fit(self.X_train, self.y_train)
                pred = model.predict(self.X_test)
                score = accuracy_score(self.y_test, pred)
                return (name, model, score)
            elif name in self.keras_model_types:
                if name == 'Keras CNN':
                    X_train_dl = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
                    X_test_dl = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                else:
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
                pred = self.predict_with_cutoffs(pred_proba)
                score = accuracy_score(self.y_test, pred)
                return (name, model, score)
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            return (name, None, 0)

    def load_data(self, X, y, feature_names=None, test_size=0.3, random_state=42):
        print("Input X shape:", X.shape)
        print("Input y shape:", y.shape if hasattr(y, 'shape') else len(y))
        print("Input y type:", type(y))

        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError("Target y must be a single column; got multiple columns.")
            y = y.iloc[:, 0]
        elif not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("Target y must be a pandas Series, DataFrame (single column), or numpy array.")

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
            self.X_train[self.categorical_columns] = self.cat_imputer.fit_transform(self.X_train[self.categorical_columns])
            self.X_test[self.categorical_columns] = self.cat_imputer.transform(self.X_test[self.categorical_columns])
            for col in self.categorical_columns:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.X_train[col] = oe.fit_transform(self.X_train[[col]])
                self.X_test[col] = oe.transform(self.X_test[[col]])
                self.label_encoders[col] = oe

        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

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

    def save_best_model(self):
        if not self.best_model:
            raise ValueError("No best model to save; train models first.")
        
        best_name, best_model, best_auc, best_ap = self.best_model
        os.makedirs(self.save_path, exist_ok=True)
        
        if best_name in self.keras_model_types:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.keras")
            best_model.model_.save(model_path)
            print(f"Saved Keras model to {model_path}")
        else:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.joblib")
            dump(best_model, model_path)
            print(f"Saved scikit-learn model to {model_path}")

    def load_best_model(self, model_name=None):
        if model_name is None and self.best_model:
            model_name = self.best_model[0]
        if not model_name:
            raise ValueError("No model name provided and no best model set.")

        if model_name in self.keras_model_types:
            model_path = os.path.join(self.save_path, f"{model_name}_best_model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Keras model not found at {model_path}")
            keras_model = load_model(model_path)
            loaded_model = KerasClassifier(model=keras_model, epochs=50, batch_size=32, verbose=0)
            print(f"Loaded Keras model from {model_path}")
        else:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Scikit-learn model not found at {model_path}")
            loaded_model = load(model_path)
            print(f"Loaded scikit-learn model from {model_path}")

        return loaded_model

    def train_and_evaluate(self):
        if self.y_train is None:
            raise ValueError("y_train is None; ensure load_data is called correctly")

        results = {}
        sklearn_models = [(name, self.all_models[name]) for name in self.models_to_train if name in self.all_models]
        keras_models = [(name, KerasClassifier(
            model=self.build_deep_learning_model(name),
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

                    X_test_input = self.X_test
                    y_pred_proba = model.predict_proba(X_test_input)
                    print(f"{name} y_pred_proba shape: {y_pred_proba.shape}")
                    
                    if self.num_classes == 2:
                        auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                        average_precision = average_precision_score(self.y_test, y_pred_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                        average_precision = average_precision_score(self.y_test, y_pred_proba, average='macro')
                    print(f"{name} AUC: {auc_score:.4f}, Average Precision: {average_precision:.4f}")

                    if auc_score > self.best_score:
                        self.best_score = auc_score
                        self.best_model = (name, model, auc_score, average_precision)

        for name, model in keras_models:
            trained_result = self._train_model(name, model)
            name, model, score = trained_result
            if model is not None:
                self.trained_models[name] = model
                results[name] = score
                print(f"{name} Test Accuracy: {score:.4f}")

                X_test_input = self.X_test if name != 'Keras CNN' else self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                y_pred_proba = model.predict_proba(X_test_input)
                print(f"{name} y_pred_proba shape: {y_pred_proba.shape}")
                
                if self.num_classes == 2:
                    auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                    average_precision = average_precision_score(self.y_test, y_pred_proba[:, 1])
                else:
                    auc_score = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                    average_precision = average_precision_score(self.y_test, y_pred_proba, average='macro')
                print(f"{name} AUC: {auc_score:.4f}, Average Precision: {average_precision:.4f}")

                if auc_score > self.best_score:
                    self.best_score = auc_score
                    self.best_model = (name, model, auc_score, average_precision)

        if self.best_model:
            best_name, best_model, best_auc, best_ap = self.best_model
            print(f"\nBest Model: {best_name} with AUC: {best_auc:.4f} and Average Precision: {best_ap:.4f}")

            X_test_input = self.X_test if best_name != 'Keras CNN' else self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
            y_pred_proba = best_model.predict_proba(X_test_input)
            self.optimal_cutoffs = self.find_optimal_cutoff(self.y_test, y_pred_proba)
            print(f"Optimal Cutoffs: {self.optimal_cutoffs}")

            self.display_target_distribution()
            self.show_confusion_matrix(y_pred_proba)
            if self.num_classes == 2:
                self.plot_roc_curve(y_pred_proba)
                self.plot_precision_recall_curve(y_pred_proba)
            else:
                self.plot_multiclass_roc_curve(self.y_test, y_pred_proba)
                self.plot_multiclass_precision_recall_curve(self.y_test, y_pred_proba)
            self.compute_dams(y_pred_proba)
            self.show_feature_importance()

            self.save_best_model()

        return results

    def build_deep_learning_model(self, model_type):
        if model_type == 'Keras MLP':
            def create_model():
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(self.num_classes if self.num_classes > 2 else 1,
                          activation='softmax' if self.num_classes > 2 else 'sigmoid')
                ])
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
                              metrics=['accuracy'])
                return model
        elif model_type == 'Keras CNN':
            def create_model():
                model = Sequential([
                    Conv1D(32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(self.num_classes if self.num_classes > 2 else 1,
                          activation='softmax' if self.num_classes > 2 else 'sigmoid')
                ])
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
                              metrics=['accuracy'])
                return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return create_model

    def find_optimal_cutoff(self, y_true, y_pred_proba):
        if self.num_classes == 2:
            pos_prob = y_pred_proba[:, 1]
            fpr, tpr, thresholds = roc_curve(y_true, pos_prob)
            youden_j = tpr + (1 - fpr) - 1
            optimal_idx = np.argmax(youden_j)
            return [thresholds[optimal_idx]]
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(self.num_classes))
            cutoffs = []
            for i in range(self.num_classes):
                fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                youden_j = tpr + (1 - fpr) - 1
                optimal_idx = np.argmax(youden_j)
                cutoffs.append(thresholds[optimal_idx])
            return cutoffs

    def predict_with_cutoffs(self, y_pred_proba):
        if self.num_classes == 2:
            return (y_pred_proba[:, 1] >= self.optimal_cutoffs[0]).astype(int)
        else:
            pred = np.zeros(y_pred_proba.shape[0], dtype=int)
            for i in range(self.num_classes):
                pred[y_pred_proba[:, i] >= self.optimal_cutoffs[i]] = i
            # Handle cases where no probability exceeds its cutoff by assigning the highest probability
            undecided = np.all(y_pred_proba < np.array(self.optimal_cutoffs), axis=1)
            pred[undecided] = np.argmax(y_pred_proba[undecided], axis=1)
            return pred

    def display_target_distribution(self):
        if self.target_is_categorical:
            y_train_display = self.target_encoder.inverse_transform(self.y_train.astype(int).reshape(-1, 1)).flatten()
        else:
            y_train_display = self.y_train

        y_series = pd.Series(y_train_display)
        total_count = len(y_series)
        
        counts = y_series.value_counts().sort_index()
        percentages = (counts / total_count * 100).round(2)
        
        dist_df = pd.DataFrame({'Class': counts.index, 'Count': counts.values, 'Percentage': percentages.values})
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Class', y='Count', data=dist_df, palette='Blues_d')
        
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.annotate(f'{int(height)}\n({dist_df["Percentage"].iloc[i]}%)',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        
        plt.title("Distribution of Target Outcome in Training Set")
        plt.xlabel("Target Class")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    def show_confusion_matrix(self, y_pred_proba):
        best_name, best_model, best_auc, best_ap = self.best_model
        y_pred = self.predict_with_cutoffs(y_pred_proba)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = self.target_encoder.categories_[0] if self.target_is_categorical else np.unique(self.y_train).astype(str)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix for {best_name} (Optimal Cutoffs: {self.optimal_cutoffs})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def get_feature_importance(self, model, model_name):
        if hasattr(model, 'feature_importances_'):
            print(f"Using native feature_importances_ for {model_name}")
            return model.feature_importances_
        elif hasattr(model, 'coef_') and model.coef_.ndim == 1:
            print(f"Using native coef_ for {model_name}")
            return np.abs(model.coef_)
        
        print(f"Computing permutation importance for {model_name}...")
        X_test_perm = self.X_test
        
        if isinstance(model, KerasClassifier) and model_name == 'Keras CNN':
            wrapped_model = KerasClassifierWrapper(model)
        else:
            wrapped_model = model
        
        perm_result = permutation_importance(wrapped_model, X_test_perm, self.y_test, n_repeats=10, random_state=42, scoring='accuracy')
        return perm_result.importances_mean
        
    def show_feature_importance(self):
        if not self.best_model:
            print("No best model selected yet.")
            return
        best_name, best_model, best_auc, best_ap = self.best_model
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
        if self.num_classes != 2:
            return
        pos_prob = y_pred_proba[:, 1]
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
        if self.num_classes != 2:
            return
        pos_prob = y_pred_proba[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, pos_prob)
        ap = average_precision_score(self.y_test, pos_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()

    def plot_multiclass_roc_curve(self, y_true, y_pred_proba):
        if self.num_classes <= 2:
            return
        y_true_binarized = label_binarize(y_true, classes=np.arange(self.num_classes))
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], label=f'{self.target_encoder.categories_[0][i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_multiclass_precision_recall_curve(self, y_true, y_pred_proba):
        if self.num_classes <= 2:
            return
        y_true_binarized = label_binarize(y_true, classes=np.arange(self.num_classes))
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            average_precision[i] = average_precision_score(y_true_binarized[:, i], y_pred_proba[:, i])
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes):
            plt.plot(recall[i], precision[i], label=f'{self.target_encoder.categories_[0][i]} (AP = {average_precision[i]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multiclass Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()

    def compute_dams(self, y_pred_proba):
        y_pred = self.predict_with_cutoffs(y_pred_proba)
        
        if self.num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            n = len(self.y_test)
            
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
    
        else:
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average=None)
            recall = recall_score(self.y_test, y_pred, average=None)
            f1 = f1_score(self.y_test, y_pred, average=None)
    
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
    
            dams_df = pd.DataFrame({
                'Class': [f'Class {i}' for i in range(self.num_classes)],
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Precision 95% CI': [f'({ci[0]:.4f}, {ci[1]:.4f})' for ci in precision_ci],
                'Recall 95% CI': [f'({ci[0]:.4f}, {ci[1]:.4f})' for ci in recall_ci],
                'F1 Score 95% CI': [f'({ci[0]:.4f}, {ci[1]:.4f})' for ci in f1_ci]
            })
    
            dams_df.loc[len(dams_df)] = ['Macro Average', 
                                          dams_df['Precision'].mean(), 
                                          dams_df['Recall'].mean(), 
                                          dams_df['F1 Score'].mean(),
                                          'N/A', 'N/A', 'N/A']
    
            print("\nDiagnostic Accuracy Measures (DAMS) for Multiclass Classification:")
            print(dams_df)

    @staticmethod
    def wilson_ci(pos, n, confidence=0.95):
        if n == 0:
            return (0, 0)
        z = norm.ppf(1 - (1 - confidence) / 2)
        phat = pos / n
        denominator = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denominator
        return (center - margin, center + margin)

    def predict(self, X_new=None):
        if not self.best_model:
            raise ValueError("No model has been trained yet.")
    
        best_name, best_model, best_auc, best_ap = self.best_model
    
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
            X_new_scaled = self.scaler.transform(X_new)
            y_true = None
    
        if best_name == 'Keras CNN':
            X_new_dl = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))
        else:
            X_new_dl = X_new_scaled
    
        pred_proba = best_model.predict_proba(X_new_dl)
        pred = self.predict_with_cutoffs(pred_proba)
    
        pred_numerical = pred.copy()
        if self.target_is_categorical and self.target_encoder is not None:
            pred = self.target_encoder.inverse_transform(pred.reshape(-1, 1)).flatten()
    
        if y_true is not None:
            accuracy = accuracy_score(y_true, pred_numerical)
            print(f"Accuracy on test set with {best_name} (Cutoffs: {self.optimal_cutoffs}): {accuracy:.4f}")
    
        return pred


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # Add OrdinalEncoder here
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, dump, load
import warnings
import multiprocessing
import os

np.random.seed(2024)
warnings.filterwarnings('ignore')

class KerasRegressorWrapper:
    """Wrapper to adjust KerasRegressor predict method for permutation importance."""
    def __init__(self, keras_regressor):
        self.model = keras_regressor

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

class MLAgentRegressor:
    def __init__(self, models_to_train=None, save_path="best_reg_model"):
        self.n_cores = multiprocessing.cpu_count()
        self.parallel_enabled = self.n_cores > 2
        self.n_jobs = max(1, self.n_cores - 1) if self.parallel_enabled else 1

        self.all_models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'Extra Trees': ExtraTreesRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'XGBoost': XGBRegressor(),
            'Sklearn ANN': MLPRegressor(max_iter=1000)
        }

        self.keras_model_types = ['Keras MLP', 'Keras CNN']
        self.models_to_train = models_to_train if models_to_train is not None else list(
            self.all_models.keys()) + self.keras_model_types
        self.best_model = None
        self.best_mae = float('inf')  # Lower MAE is better
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.num_imputer = None
        self.cat_imputer = None
        self.label_encoders = {}
        self.trained_models = {}
        self.save_path = save_path

    def _train_model(self, name, model):
        try:
            if name in self.all_models:
                model.fit(self.X_train, self.y_train)
                pred = model.predict(self.X_test)
                mae = mean_absolute_error(self.y_test, pred)
                return (name, model, mae)
            elif name in self.keras_model_types:
                if name == 'Keras CNN':
                    X_train_dl = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
                    X_test_dl = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                else:
                    X_train_dl = self.X_train
                    X_test_dl = self.X_test
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(
                    X_train_dl, self.y_train,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                pred = model.predict(X_test_dl)
                mae = mean_absolute_error(self.y_test, pred)
                return (name, model, mae)
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            return (name, None, float('inf'))

    def load_data(self, X, y, feature_names=None, test_size=0.3, random_state=42):
        print("Input X shape:", X.shape)
        print("Input y shape:", y.shape if hasattr(y, 'shape') else len(y))
        print("Input y type:", type(y))

        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError("Target y must be a single column; got multiple columns.")
            y = y.iloc[:, 0]
        elif not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("Target y must be a pandas Series, DataFrame (single column), or numpy array.")

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if self.numerical_columns:
            self.num_imputer = SimpleImputer(strategy='mean')
            self.X_train[self.numerical_columns] = self.num_imputer.fit_transform(self.X_train[self.numerical_columns])
            self.X_test[self.numerical_columns] = self.num_imputer.transform(self.X_test[self.numerical_columns])

        if self.categorical_columns:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            self.X_train[self.categorical_columns] = self.cat_imputer.fit_transform(self.X_train[self.categorical_columns])
            self.X_test[self.categorical_columns] = self.cat_imputer.transform(self.X_test[self.categorical_columns])
            for col in self.categorical_columns:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.X_train[col] = oe.fit_transform(self.X_train[[col]])
                self.X_test[col] = oe.transform(self.X_test[[col]])
                self.label_encoders[col] = oe

        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("Final X_train shape:", self.X_train.shape)
        print("Final y_train shape:", self.y_train.shape if hasattr(self.y_train, 'shape') else len(self.y_train))

    def save_best_model(self):
        if not self.best_model:
            raise ValueError("No best model to save; train models first.")
        
        best_name, best_model, best_mae = self.best_model
        os.makedirs(self.save_path, exist_ok=True)
        
        if best_name in self.keras_model_types:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.keras")
            best_model.model_.save(model_path)
            print(f"Saved Keras model to {model_path}")
        else:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.joblib")
            dump(best_model, model_path)
            print(f"Saved scikit-learn model to {model_path}")

    def load_best_model(self, model_name=None):
        if model_name is None and self.best_model:
            model_name = self.best_model[0]
        if not model_name:
            raise ValueError("No model name provided and no best model set.")

        if model_name in self.keras_model_types:
            model_path = os.path.join(self.save_path, f"{model_name}_best_model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Keras model not found at {model_path}")
            keras_model = load_model(model_path)
            loaded_model = KerasRegressor(model=keras_model, epochs=50, batch_size=32, verbose=0)
            print(f"Loaded Keras model from {model_path}")
        else:
            model_path = os.path.join(self.save_path, f"{model_name}_best_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Scikit-learn model not found at {model_path}")
            loaded_model = load(model_path)
            print(f"Loaded scikit-learn model from {model_path}")

        return loaded_model

    def train_and_evaluate(self):
        if self.y_train is None:
            raise ValueError("y_train is None; ensure load_data is called correctly")

        results = {}
        sklearn_models = [(name, self.all_models[name]) for name in self.models_to_train if name in self.all_models]
        keras_models = [(name, KerasRegressor(
            model=self.build_deep_learning_model(name),
            epochs=50,
            batch_size=32,
            verbose=0
        )) for name in self.models_to_train if name in self.keras_model_types]

        if sklearn_models:
            trained_sklearn = Parallel(n_jobs=self.n_jobs)(
                delayed(self._train_model)(name, model) for name, model in sklearn_models
            )
            for name, model, mae in trained_sklearn:
                if model is not None:
                    self.trained_models[name] = model
                    results[name] = mae
                    pred = model.predict(self.X_test)
                    mse = mean_squared_error(self.y_test, pred)
                    r2 = r2_score(self.y_test, pred)
                    print(f"{name} MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
                    if mae < self.best_mae:
                        self.best_mae = mae
                        self.best_model = (name, model, mae)

        for name, model in keras_models:
            trained_result = self._train_model(name, model)
            name, model, mae = trained_result
            if model is not None:
                self.trained_models[name] = model
                results[name] = mae
                X_test_input = self.X_test if name != 'Keras CNN' else self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                pred = model.predict(X_test_input)
                mse = mean_squared_error(self.y_test, pred)
                r2 = r2_score(self.y_test, pred)
                print(f"{name} MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
                if mae < self.best_mae:
                    self.best_mae = mae
                    self.best_model = (name, model, mae)

        if self.best_model:
            best_name, best_model, best_mae = self.best_model
            print(f"\nBest Model: {best_name} with MAE: {best_mae:.4f}")
            X_test_input = self.X_test if best_name != 'Keras CNN' else self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
            y_pred = best_model.predict(X_test_input)
            self.display_target_distribution()
            self.plot_prediction_vs_actual(y_pred)
            self.compute_regression_metrics(y_pred)
            self.show_feature_importance()
            self.save_best_model()

        return results

    def build_deep_learning_model(self, model_type):
        if model_type == 'Keras MLP':
            def create_model():
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1)  # Single output for regression
                ])
                model.compile(optimizer='adam', loss='mae', metrics=['mae'])
                return model
        elif model_type == 'Keras CNN':
            def create_model():
                model = Sequential([
                    Conv1D(32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1)  # Single output for regression
                ])
                model.compile(optimizer='adam', loss='mae', metrics=['mae'])
                return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return create_model

    def display_target_distribution(self):
        y_series = pd.Series(self.y_train)
        total_count = len(y_series)
        plt.figure(figsize=(10, 6))
        sns.histplot(y_series, bins=30, kde=True, color='blue')
        plt.title("Distribution of Target Variable in Training Set")
        plt.xlabel("Target Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_prediction_vs_actual(self, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        plt.tight_layout()
        plt.show()

    def compute_regression_metrics(self, y_pred):
        # Calculate point estimates
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Bootstrap parameters
        n_iterations = 1000
        n_size = len(self.y_test)
        bootstrap_mae = []
        bootstrap_mse = []
        bootstrap_rmse = []
        bootstrap_r2 = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Perform bootstrapping
        for _ in range(n_iterations):
            # Generate bootstrap sample indices
            indices = np.random.choice(n_size, size=n_size, replace=True)
            y_test_sample = self.y_test[indices]
            y_pred_sample = y_pred[indices]
            
            # Calculate metrics for bootstrap sample
            boot_mae = mean_absolute_error(y_test_sample, y_pred_sample)
            boot_mse = mean_squared_error(y_test_sample, y_pred_sample)
            boot_rmse = np.sqrt(boot_mse)
            boot_r2 = r2_score(y_test_sample, y_pred_sample)
            
            bootstrap_mae.append(boot_mae)
            bootstrap_mse.append(boot_mse)
            bootstrap_rmse.append(boot_rmse)
            bootstrap_r2.append(boot_r2)
        
        # Calculate 95% confidence intervals (2.5th and 97.5th percentiles)
        alpha = 0.05
        mae_ci = np.percentile(bootstrap_mae, [100 * alpha/2, 100 * (1 - alpha/2)])
        mse_ci = np.percentile(bootstrap_mse, [100 * alpha/2, 100 * (1 - alpha/2)])
        rmse_ci = np.percentile(bootstrap_rmse, [100 * alpha/2, 100 * (1 - alpha/2)])
        r2_ci = np.percentile(bootstrap_r2, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        # Create results DataFrame with point estimates and CIs
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
            'Value': [mae, mse, rmse, r2],
            '95% CI Lower': [mae_ci[0], mse_ci[0], rmse_ci[0], r2_ci[0]],
            '95% CI Upper': [mae_ci[1], mse_ci[1], rmse_ci[1], r2_ci[1]]
        })
        
        # Format the output for better readability
        metrics_df['95% CI'] = metrics_df.apply(
            lambda row: f"[{row['95% CI Lower']:.4f}, {row['95% CI Upper']:.4f}]", axis=1
        )
        
        print("\nRegression Metrics with 95% Confidence Intervals:")
        print(metrics_df[['Metric', 'Value', '95% CI']].to_string(index=False))
        
        return metrics_df
        
    def get_feature_importance(self, model, model_name):
        if hasattr(model, 'feature_importances_'):
            print(f"Using native feature_importances_ for {model_name}")
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            print(f"Using native coef_ for {model_name}")
            return np.abs(model.coef_)
        
        print(f"Computing permutation importance for {model_name}...")
        X_test_perm = self.X_test
        
        if model_name == 'Keras CNN':
            # Reshape to 3D for prediction
            X_test_perm_3d = X_test_perm.reshape(X_test_perm.shape[0], X_test_perm.shape[1], 1)
            # Flatten to 2D for permutation_importance
            X_test_perm_2d = X_test_perm_3d.reshape(X_test_perm_3d.shape[0], -1)
            
            class KerasCNNWrapper:
                def __init__(self, model):
                    self.model = model
                    self.keras_model = model.model_
                
                def predict(self, X):
                    # Reshape flat 2D input back to 3D for CNN
                    if len(X.shape) == 2:
                        X = X.reshape(X.shape[0], self.keras_model.input_shape[1], 1)
                    return self.model.predict(X)
                
                def fit(self, X, y, **kwargs):
                    return self.model.fit(X, y, **kwargs)
            
            wrapped_model = KerasCNNWrapper(model)
            perm_result = permutation_importance(
                wrapped_model,
                X_test_perm_2d,
                self.y_test,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_absolute_error'
            )
        else:
            X_test_perm_2d = X_test_perm
            
            if isinstance(model, KerasRegressor):
                class KerasWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.is_cnn = model_name == 'Keras CNN'
                        self.keras_model = model.model_ if self.is_cnn else None
                    
                    def predict(self, X):
                        if self.is_cnn and len(X.shape) == 2:
                            X = X.reshape(X.shape[0], self.keras_model.input_shape[1], 1)
                        return self.model.predict(X)
                    
                    def fit(self, X, y, **kwargs):
                        return self.model.fit(X, y, **kwargs)
                
                wrapped_model = KerasWrapper(model)
            else:
                wrapped_model = model
                
            perm_result = permutation_importance(
                wrapped_model,
                X_test_perm_2d,
                self.y_test,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_absolute_error'
            )
        
        return perm_result.importances_mean
        
    def show_feature_importance(self):
        if not self.best_model:
            print("No best model selected yet.")
            return
        best_name, best_model, best_mae = self.best_model
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

    def predict(self, X_new=None):
        if not self.best_model:
            raise ValueError("No model has been trained yet.")

        best_name, best_model, best_mae = self.best_model

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
            X_new_scaled = self.scaler.transform(X_new)
            y_true = None

        if best_name == 'Keras CNN':
            X_new_dl = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))
        else:
            X_new_dl = X_new_scaled

        pred = best_model.predict(X_new_dl)

        if y_true is not None:
            mae = mean_absolute_error(y_true, pred)
            print(f"MAE on test set with {best_name}: {mae:.4f}")

        return pred


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.discrete.truncated_model import HurdleCountModel
from pygam import PoissonGAM
from catboost import CatBoostRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, dump, load
import warnings
import multiprocessing
import os

np.random.seed(2024)
warnings.filterwarnings('ignore')

class MLAgentCountRegressor:
    def __init__(self, models_to_train=None, save_path="best_count_model"):
        self.n_cores = multiprocessing.cpu_count()
        self.parallel_enabled = self.n_cores > 2
        self.n_jobs = max(1, self.n_cores - 1) if self.parallel_enabled else 1

        self.all_models = {
            'Poisson Regression': PoissonRegressor(),
            'Negative Binomial': None,
            'GLM Log-Link': None,
            'Zero-Inflated Poisson': None,
            'Hurdle Model': None,
            'Zero-Inflated 0-1': None,
            'GAM Poisson': PoissonGAM(),
            'CatBoost Poisson': CatBoostRegressor(loss_function='Poisson', verbose=0),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(loss='poisson'),
            'Extra Trees': ExtraTreesRegressor(),
            'XGBoost': xgb.XGBRegressor(objective='count:poisson')
        }

        self.keras_model_types = ['Keras MLP', 'Keras CNN', 'Keras RNN', 'Keras VAE']
        self.models_to_train = models_to_train if models_to_train is not None else list(
            self.all_models.keys()) + self.keras_model_types
        self.best_model = None
        self.best_poisson_dev = float('inf')  # Changed from best_mae to best_poisson_dev
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.num_imputer = None
        self.cat_imputer = None
        self.label_encoders = {}
        self.trained_models = {}
        self.save_path = save_path

    @staticmethod
    def poisson_deviance(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_pred = np.clip(y_pred, 1e-10, None)
        dev = np.zeros_like(y_true)
        mask = y_true > 0
        dev[mask] = 2 * (y_true[mask] * np.log(y_true[mask] / y_pred[mask]) - (y_true[mask] - y_pred[mask]))
        dev[~mask] = 2 * y_pred[~mask]
        return np.mean(dev)

    def _train_model(self, name, model):
        try:
            if name in ['Poisson Regression', 'Linear Regression', 'Decision Tree', 
                       'Random Forest', 'Gradient Boosting', 'Extra Trees', 'CatBoost Poisson', 
                       'GAM Poisson', 'XGBoost']:
                model.fit(self.X_train, self.y_train)
                pred = model.predict(self.X_test)
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, model, poisson_dev)
            elif name == 'Negative Binomial':
                X_train_const = sm.add_constant(self.X_train)
                X_test_const = sm.add_constant(self.X_test)
                model = sm.GLM(self.y_train, X_train_const, family=sm.families.NegativeBinomial()).fit()
                pred = model.predict(X_test_const)
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, model, poisson_dev)
            elif name == 'GLM Log-Link':
                X_train_const = sm.add_constant(self.X_train)
                X_test_const = sm.add_constant(self.X_test)
                model = sm.GLM(self.y_train, X_train_const, family=sm.families.Poisson(link=sm.families.links.log())).fit()
                pred = model.predict(X_test_const)
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, model, poisson_dev)
            elif name == 'Zero-Inflated Poisson':
                X_train_const = sm.add_constant(self.X_train)
                X_test_const = sm.add_constant(self.X_test)
                model = ZeroInflatedPoisson(self.y_train, X_train_const).fit()
                pred = model.predict(X_test_const)
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, model, poisson_dev)
            elif name == 'Hurdle Model':
                X_train_const = sm.add_constant(self.X_train)
                X_test_const = sm.add_constant(self.X_test)
                model = HurdleCountModel(self.y_train, X_train_const, dist='poisson').fit()
                pred = model.predict(X_test_const)
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, model, poisson_dev)
            elif name == 'Zero-Inflated 0-1':
                X_train_const = sm.add_constant(self.X_train)
                X_test_const = sm.add_constant(self.X_test)
                binary_y = (self.y_train > 0).astype(int)
                count_y = self.y_train[self.y_train > 0]
                X_train_count = self.X_train[self.y_train > 0]
                X_train_count_const = sm.add_constant(X_train_count)
                
                zero_model = sm.Logit(binary_y, X_train_const).fit(disp=0)
                count_model = sm.GLM(count_y, X_train_count_const, family=sm.families.Poisson()).fit()
                
                zero_pred = zero_model.predict(X_test_const)
                count_pred = count_model.predict(X_test_const)
                pred = zero_pred * count_pred
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, (zero_model, count_model), poisson_dev)
            elif name in self.keras_model_types:
                if name in ['Keras CNN', 'Keras RNN', 'Keras VAE']:
                    X_train_dl = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
                    X_test_dl = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                else:
                    X_train_dl = self.X_train
                    X_test_dl = self.X_test
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(
                    X_train_dl if name != 'Keras VAE' else [X_train_dl, self.y_train],
                    self.y_train,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                pred = model.predict(X_test_dl if name != 'Keras VAE' else [X_test_dl])
                if name == 'Keras VAE':
                    pred = pred[0]  # Assuming the first output is the prediction
                pred = np.round(pred).clip(min=0)
                poisson_dev = self.poisson_deviance(self.y_test, pred)
                return (name, model, poisson_dev)
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            return (name, None, float('inf'))

    def load_data(self, X, y, feature_names=None, test_size=0.3, random_state=42):
        print("Input X shape:", X.shape)
        print("Input y shape:", y.shape if hasattr(y, 'shape') else len(y))
        print("Input y type:", type(y))

        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError("Target y must be a single column; got multiple columns.")
            y = y.iloc[:, 0]
        elif not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("Target y must be a pandas Series, DataFrame (single column), or numpy array.")

        if not np.all(y >= 0) or not np.all(y == np.round(y)):
            raise ValueError("Target y must contain non-negative integers for count data.")

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if self.numerical_columns:
            self.num_imputer = SimpleImputer(strategy='mean')
            self.X_train[self.numerical_columns] = self.num_imputer.fit_transform(self.X_train[self.numerical_columns])
            self.X_test[self.numerical_columns] = self.num_imputer.transform(self.X_test[self.numerical_columns])

        if self.categorical_columns:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            self.X_train[self.categorical_columns] = self.cat_imputer.fit_transform(self.X_train[self.categorical_columns])
            self.X_test[self.categorical_columns] = self.cat_imputer.transform(self.X_test[self.categorical_columns])
            for col in self.categorical_columns:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.X_train[col] = oe.fit_transform(self.X_train[[col]])
                self.X_test[col] = oe.transform(self.X_test[[col]])
                self.label_encoders[col] = oe

        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.y_train = np.array(self.y_train, dtype=int)
        self.y_test = np.array(self.y_test, dtype=int)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("Final X_train shape:", self.X_train.shape)
        print("Final y_train shape:", self.y_train.shape if hasattr(self.y_train, 'shape') else len(self.y_train))

    def save_best_model(self):
        if not self.best_model:
            raise ValueError("No best model to save; train models first.")
        
        best_name, best_model, best_poisson_dev = self.best_model  # Updated from best_mae
        os.makedirs(self.save_path, exist_ok=True)
        
        if best_name in self.keras_model_types:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.keras")
            best_model.model_.save(model_path)
            print(f"Saved Keras model to {model_path}")
        elif best_name == 'Zero-Inflated 0-1':
            zero_model, count_model = best_model
            zero_path = os.path.join(self.save_path, f"{best_name}_zero_model.joblib")
            count_path = os.path.join(self.save_path, f"{best_name}_count_model.joblib")
            dump(zero_model, zero_path)
            dump(count_model, count_path)
            print(f"Saved Zero-Inflated 0-1 models to {zero_path} and {count_path}")
        else:
            model_path = os.path.join(self.save_path, f"{best_name}_best_model.joblib")
            dump(best_model, model_path)
            print(f"Saved model to {model_path}")

    def train_and_evaluate(self):
        if self.y_train is None:
            raise ValueError("y_train is None; ensure load_data is called correctly")

        results = {}
        sklearn_models = [(name, self.all_models[name]) for name in self.models_to_train 
                          if name in ['Poisson Regression', 'Linear Regression', 'Decision Tree', 
                                      'Random Forest', 'Gradient Boosting', 'Extra Trees', 
                                      'GAM Poisson', 'CatBoost Poisson', 'XGBoost']]
        statsmodels_models = [(name, None) for name in self.models_to_train 
                              if name in ['Negative Binomial', 'GLM Log-Link', 'Zero-Inflated Poisson', 
                                         'Hurdle Model', 'Zero-Inflated 0-1']]
        keras_models = [(name, KerasRegressor(
            model=self.build_deep_learning_model(name),
            epochs=50,
            batch_size=32,
            verbose=0
        )) for name in self.models_to_train if name in self.keras_model_types]

        # Train sklearn-compatible models in parallel
        if sklearn_models:
            trained_sklearn = Parallel(n_jobs=self.n_jobs)(
                delayed(self._train_model)(name, model) for name, model in sklearn_models
            )
            for name, model, poisson_dev in trained_sklearn:
                if model is not None:
                    self.trained_models[name] = model
                    results[name] = poisson_dev
                    print(f"{name} Poisson Deviance: {poisson_dev:.4f}")
                    if poisson_dev < self.best_poisson_dev:
                        self.best_poisson_dev = poisson_dev
                        self.best_model = (name, model, poisson_dev)

        # Train statsmodels models sequentially
        for name, _ in statsmodels_models:
            trained_result = self._train_model(name, None)
            name, model, poisson_dev = trained_result
            if model is not None:
                self.trained_models[name] = model
                results[name] = poisson_dev
                print(f"{name} Poisson Deviance: {poisson_dev:.4f}")
                if poisson_dev < self.best_poisson_dev:
                    self.best_poisson_dev = poisson_dev
                    self.best_model = (name, model, poisson_dev)

        # Train Keras models sequentially
        for name, model in keras_models:
            trained_result = self._train_model(name, model)
            name, model, poisson_dev = trained_result
            if model is not None:
                self.trained_models[name] = model
                results[name] = poisson_dev
                print(f"{name} Poisson Deviance: {poisson_dev:.4f}")
                if poisson_dev < self.best_poisson_dev:
                    self.best_poisson_dev = poisson_dev
                    self.best_model = (name, model, poisson_dev)

        if self.best_model:
            best_name, best_model, best_poisson_dev = self.best_model
            print(f"\nBest Model: {best_name} with Poisson Deviance: {best_poisson_dev:.4f}")
            if best_name in self.keras_model_types:
                X_test_input = self.X_test if best_name == 'Keras MLP' else self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
                y_pred = np.round(best_model.predict(X_test_input)).clip(min=0)
                if best_name == 'Keras VAE':
                    y_pred = y_pred[0]  # Assuming the first output is the prediction
            elif best_name == 'Zero-Inflated 0-1':
                zero_model, count_model = best_model
                X_test_const = sm.add_constant(self.X_test)
                zero_pred = zero_model.predict(X_test_const)
                count_pred = count_model.predict(X_test_const)
                y_pred = np.round(zero_pred * count_pred).clip(min=0)
            else:
                X_input = sm.add_constant(self.X_test) if best_name in ['Negative Binomial', 'GLM Log-Link', 'Zero-Inflated Poisson', 'Hurdle Model'] else self.X_test
                y_pred = np.round(best_model.predict(X_input)).clip(min=0)
            self.display_target_distribution()
            self.plot_prediction_vs_actual(y_pred)
            self.compute_regression_metrics(y_pred)
            self.show_feature_importance()
            self.save_best_model()

        return results

    def build_deep_learning_model(self, model_type):
        if model_type == 'Keras MLP':
            def create_model():
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='softplus')
                ])
                model.compile(optimizer='adam', loss='poisson', metrics=['mae'])
                return model
        elif model_type == 'Keras CNN':
            def create_model():
                model = Sequential([
                    Conv1D(32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='softplus')
                ])
                model.compile(optimizer='adam', loss='poisson', metrics=['mae'])
                return model
        elif model_type == 'Keras RNN':
            def create_model():
                model = Sequential([
                    LSTM(32, activation='relu', input_shape=(self.X_train.shape[1], 1), return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='softplus')
                ])
                model.compile(optimizer='adam', loss='poisson', metrics=['mae'])
                return model
        elif model_type == 'Keras VAE':
            latent_dim = 2
            def create_model():
                inputs = Input(shape=(self.X_train.shape[1], 1))
                x = Conv1D(16, kernel_size=2, activation='relu', padding='same')(inputs)
                x = Flatten()(x)
                z_mean = Dense(latent_dim)(x)
                z_log_var = Dense(latent_dim)(x)
                
                def sampling(args):
                    z_mean, z_log_var = args
                    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
                    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
                
                z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
                x = Dense(16, activation='relu')(z)
                outputs = Dense(1, activation='softplus')(x)
                
                model = Model(inputs, outputs)
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.poisson(inputs, outputs))
                kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                total_loss = reconstruction_loss + kl_loss
                model.add_loss(total_loss)
                model.compile(optimizer='adam', metrics=['mae'])
                return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return create_model

    def display_target_distribution(self):
        y_series = pd.Series(self.y_train)
        plt.figure(figsize=(10, 6))
        sns.histplot(y_series, bins=30, kde=False, color='blue', stat='count')
        plt.title("Distribution of Count Target Variable in Training Set")
        plt.xlabel("Count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_prediction_vs_actual(self, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Counts")
        plt.ylabel("Predicted Counts")
        plt.title("Predicted vs Actual Counts")
        plt.tight_layout()
        plt.show()

    def compute_regression_metrics(self, y_pred):
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        n_iterations = 1000
        n_size = len(self.y_test)
        bootstrap_mae = []
        bootstrap_mse = []
        bootstrap_rmse = []
        bootstrap_r2 = []
        
        np.random.seed(42)
        
        for _ in range(n_iterations):
            indices = np.random.choice(n_size, size=n_size, replace=True)
            y_test_sample = self.y_test[indices]
            y_pred_sample = y_pred[indices]
            
            boot_mae = mean_absolute_error(y_test_sample, y_pred_sample)
            boot_mse = mean_squared_error(y_test_sample, y_pred_sample)
            boot_rmse = np.sqrt(boot_mse)
            boot_r2 = r2_score(y_test_sample, y_pred_sample)
            
            bootstrap_mae.append(boot_mae)
            bootstrap_mse.append(boot_mse)
            bootstrap_rmse.append(boot_rmse)
            bootstrap_r2.append(boot_r2)
        
        alpha = 0.05
        mae_ci = np.percentile(bootstrap_mae, [100 * alpha/2, 100 * (1 - alpha/2)])
        mse_ci = np.percentile(bootstrap_mse, [100 * alpha/2, 100 * (1 - alpha/2)])
        rmse_ci = np.percentile(bootstrap_rmse, [100 * alpha/2, 100 * (1 - alpha/2)])
        r2_ci = np.percentile(bootstrap_r2, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
            'Value': [mae, mse, rmse, r2],
            '95% CI Lower': [mae_ci[0], mse_ci[0], rmse_ci[0], r2_ci[0]],
            '95% CI Upper': [mae_ci[1], mse_ci[1], rmse_ci[1], r2_ci[1]]
        })
        
        metrics_df['95% CI'] = metrics_df.apply(
            lambda row: f"[{row['95% CI Lower']:.4f}, {row['95% CI Upper']:.4f}]", axis=1
        )
        
        print("\nRegression Metrics with 95% Confidence Intervals for Count Data:")
        print(metrics_df[['Metric', 'Value', '95% CI']].to_string(index=False))
        
        return metrics_df

    def get_feature_importance(self, model, model_name):
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        elif model_name in ['Negative Binomial', 'GLM Log-Link', 'Zero-Inflated Poisson', 'Hurdle Model']:
            return np.abs(model.params[1:])
        elif model_name == 'Zero-Inflated 0-1':
            zero_model, count_model = model
            return np.abs(count_model.params[1:])
        elif model_name == 'GAM Poisson':
            return np.abs(model.coef_)
        
        X_test_perm = self.X_test
        if model_name in ['Keras CNN', 'Keras RNN', 'Keras VAE']:
            X_test_perm_3d = X_test_perm.reshape(X_test_perm.shape[0], X_test_perm.shape[1], 1)
            X_test_perm_2d = X_test_perm_3d.reshape(X_test_perm_3d.shape[0], -1)
            
            class KerasCountWrapper:
                def __init__(self, model, is_vae=False):
                    self.model = model
                    self.is_vae = is_vae
                
                def predict(self, X):
                    if len(X.shape) == 2:
                        X = X.reshape(X.shape[0], X.shape[1] // (1 if not self.is_vae else 1), 1)
                    pred = self.model.predict([X])[0] if self.is_vae else self.model.predict(X)
                    return np.round(pred).clip(min=0)
                
                def fit(self, X, y, **kwargs):
                    return self.model.fit(X, y, **kwargs)
            
            wrapped_model = KerasCountWrapper(model, is_vae=(model_name == 'Keras VAE'))
            perm_result = permutation_importance(
                wrapped_model,
                X_test_perm_2d,
                self.y_test,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_absolute_error'
            )
        else:
            X_test_perm_2d = X_test_perm
            
            if model_name in ['Negative Binomial', 'GLM Log-Link', 'Zero-Inflated Poisson', 'Hurdle Model']:
                class StatsCountWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, X):
                        pred = self.model.predict(sm.add_constant(X))
                        return np.round(pred).clip(min=0)
                
                wrapped_model = StatsCountWrapper(model)
            elif model_name == 'Zero-Inflated 0-1':
                zero_model, count_model = model
                class ZeroOneWrapper:
                    def __init__(self, zero_model, count_model):
                        self.zero_model = zero_model
                        self.count_model = count_model
                    
                    def predict(self, X):
                        zero_pred = self.zero_model.predict(sm.add_constant(X))
                        count_pred = self.count_model.predict(sm.add_constant(X))
                        return np.round(zero_pred * count_pred).clip(min=0)
                
                wrapped_model = ZeroOneWrapper(zero_model, count_model)
            else:
                wrapped_model = model
                
            perm_result = permutation_importance(
                wrapped_model,
                X_test_perm_2d,
                self.y_test,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_absolute_error'
            )
        
        return perm_result.importances_mean

    def show_feature_importance(self):
        if not self.best_model:
            print("No best model selected yet.")
            return
        best_name, best_model, best_poisson_dev = self.best_model  # Updated from best_mae
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

    def predict(self, X_new=None):
        if not self.best_model:
            raise ValueError("No model has been trained yet.")

        best_name, best_model, best_poisson_dev = self.best_model  # Updated from best_mae

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
            X_new_scaled = self.scaler.transform(X_new)
            y_true = None

        if best_name in ['Keras CNN', 'Keras RNN', 'Keras VAE']:
            X_new_dl = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))
            pred = np.round(best_model.predict(X_new_dl)).clip(min=0)
            if best_name == 'Keras VAE':
                pred = pred[0]  # Assuming the first output is the prediction
        elif best_name == 'Keras MLP':
            pred = np.round(best_model.predict(X_new_scaled)).clip(min=0)
        elif best_name == 'Zero-Inflated 0-1':
            zero_model, count_model = best_model
            X_new_const = sm.add_constant(X_new_scaled)
            zero_pred = zero_model.predict(X_new_const)
            count_pred = count_model.predict(X_new_const)
            pred = np.round(zero_pred * count_pred).clip(min=0)
        else:
            X_input = sm.add_constant(X_new_scaled) if best_name in ['Negative Binomial', 'GLM Log-Link', 'Zero-Inflated Poisson', 'Hurdle Model'] else X_new_scaled
            pred = np.round(best_model.predict(X_input)).clip(min=0)

        if y_true is not None:
            poisson_dev = self.poisson_deviance(y_true, pred)  # Updated to report Poisson deviance
            print(f"Poisson Deviance on test set with {best_name}: {poisson_dev:.4f}")

        return pred

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from hmmlearn.hmm import GaussianHMM
import gpboost as gpb
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings

# Suppress warnings
os.environ["OMP_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

# Existing model classes (unchanged for brevity, include previous fixes)
class REEMLogisticClassifier:
    def __init__(self, max_iter=100, tol=1e-4, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_effects_ = None
        self.fixed_model_ = None
        self.scaler_ = StandardScaler()
        self.n_classes_ = None

    def _initialize_random_effects(self, ids, n_classes):
        return defaultdict(lambda: np.zeros(n_classes))

    def _e_step(self, fixed_effects_pred, y, ids):
        new_random_effects = defaultdict(lambda: np.zeros(self.n_classes_))
        y_one_hot = np.eye(self.n_classes_)[y]
        for i in np.unique(ids):
            idx = (ids == i)
            y_i = y_one_hot[idx]
            fixed_pred_i = fixed_effects_pred[idx]

            def log_likelihood(bi):
                logits = np.log(fixed_pred_i + 1e-10) + bi
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                return -np.mean(np.sum(y_i * np.log(probs + 1e-10), axis=1))

            best_bi, best_loss = np.zeros(self.n_classes_), np.inf
            for bi_shift in np.linspace(-1, 1, 11):
                bi = np.zeros(self.n_classes_)
                bi += bi_shift
                l = log_likelihood(bi)
                if l < best_loss:
                    best_loss, best_bi = l, bi.copy()
            new_random_effects[i] = best_bi
        return new_random_effects

    def _m_step(self, X, y, random_effects, ids):
        model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        model.fit(X, y)
        return model

    def fit(self, X, y, ids):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            ids_flat = np.concatenate([np.full(x.shape[0], id_val) for x, id_val in zip(X, ids)])
        else:
            X_flat = X
            ids_flat = ids

        self.n_classes_ = len(np.unique(y))
        X_scaled = self.scaler_.fit_transform(X_flat)
        self.random_effects_ = self._initialize_random_effects(ids_flat, self.n_classes_)

        for iteration in tqdm(range(self.max_iter), desc="REEM Logistic Fitting", disable=not self.verbose):
            self.fixed_model_ = self._m_step(X_scaled, y, self.random_effects_, ids_flat)
            fixed_preds = self.fixed_model_.predict_proba(X_scaled)
            new_random_effects = self._e_step(fixed_preds, y, ids_flat)

            diffs = np.array([
                np.max(np.abs(self.random_effects_[i] - new_random_effects[i]))
                for i in new_random_effects if i in self.random_effects_
            ])
            self.random_effects_ = new_random_effects
            if len(diffs) > 0 and np.max(diffs) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        return self

    def predict_proba(self, X, ids):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]

        X_scaled = self.scaler_.transform(X_flat)
        probs_flat = self.fixed_model_.predict_proba(X_scaled)
        for idx, i in enumerate(ids):
            logits = np.log(probs_flat[idx] + 1e-10) + self.random_effects_.get(i, np.zeros(self.n_classes_))
            probs_flat[idx] = np.exp(logits) / np.sum(np.exp(logits))

        probs = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            probs.append(probs_flat[start:end][-1])
            start = end
        return np.array(probs)

    def predict(self, X, ids):
        probs = self.predict_proba(X, ids)
        return np.argmax(probs, axis=1)

    def score(self, X, y, ids):
        probs = self.predict_proba(X, ids)
        return auc_roc_score(y, probs, self.n_classes_)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.random_effects_ is not None:
            state['random_effects_'] = dict(self.random_effects_)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'random_effects_' in state and state['random_effects_'] is not None:
            self.random_effects_ = defaultdict(lambda: np.zeros(self.n_classes_), state['random_effects_'])

class LongitudinalRFClassifier:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        probs_flat = self.model.predict_proba(X_scaled)
        probs = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            probs.append(probs_flat[start:end][-1])
            start = end
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        probs = self.predict_proba(X)
        return auc_roc_score(y, probs, len(np.unique(y)))

class HMMClassifier:
    def __init__(self, n_components=2):
        self.model = GaussianHMM(n_components=n_components)
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        states = self.model.predict(X_scaled)
        self.classifier.fit(states.reshape(-1, 1), y)

    def predict_proba(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        states = self.model.predict(X_scaled)
        probs_flat = self.classifier.predict_proba(states.reshape(-1, 1))
        probs = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            probs.append(probs_flat[start:end][-1])
            start = end
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        probs = self.predict_proba(X)
        return auc_roc_score(y, probs, len(np.unique(y)))

class GPBoostClassifier:
    def __init__(self):
        self.model = gpb.GPBoostClassifier(objective="multiclass")
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        probs_flat = self.model.predict_proba(X_scaled)
        probs = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            probs.append(probs_flat[start:end][-1])
            start = end
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        probs = self.predict_proba(X)
        return auc_roc_score(y, probs, len(np.unique(y)))

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, num_channels=[16, 32], kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          dilation=dilation_size, padding=(kernel_size-1)*dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        return torch.softmax(self.fc(x), dim=-1)

    def predict_proba(self, X, device='cpu'):
        """Predict probabilities for input X (list of arrays or tensor)."""
        self.eval()
        if isinstance(X, list):
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = self(X_tensor).cpu().numpy()
        return probs

    def predict(self, X, device='cpu'):
        """Predict class labels using argmax."""
        probs = self.predict_proba(X, device)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _pad_sequences(X):
        """Pad sequences for consistent length."""
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return torch.softmax(self.fc(hn[-1]), dim=-1)

    def predict_proba(self, X, device='cpu'):
        self.eval()
        if isinstance(X, list):
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = self(X_tensor).cpu().numpy()
        return probs

    def predict(self, X, device='cpu'):
        probs = self.predict_proba(X, device)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _pad_sequences(X):
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, hn = self.gru(x)
        return torch.softmax(self.fc(hn[-1]), dim=-1)

    def predict_proba(self, X, device='cpu'):
        self.eval()
        if isinstance(X, list):
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = self(X_tensor).cpu().numpy()
        return probs

    def predict(self, X, device='cpu'):
        probs = self.predict_proba(X, device)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _pad_sequences(X):
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, d_model=64, nhead=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return torch.softmax(self.fc(x[:, -1, :]), dim=-1)

    def predict_proba(self, X, device='cpu'):
        self.eval()
        if isinstance(X, list):
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = self(X_tensor).cpu().numpy()
        return probs

    def predict(self, X, device='cpu'):
        probs = self.predict_proba(X, device)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _pad_sequences(X):
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor(y, dtype=torch.long)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids[idx]

def auc_roc_score(y_true, y_prob, n_classes):
    """Compute AUC-ROC, handling binary and multiclass cases."""
    if n_classes == 2:
        return auc(roc_curve(y_true, y_prob[:, 1])[0], roc_curve(y_true, y_prob[:, 1])[1])
    else:
        # Macro-average AUC-ROC for multiclass (one-vs-rest)
        auc_scores = []
        y_one_hot = np.eye(n_classes)[y_true]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_one_hot[:, i], y_prob[:, i])
            auc_scores.append(auc(fpr, tpr))
        return np.mean(auc_scores)

def auprc_score(y_true, y_prob, n_classes):
    """Compute AUPRC, handling binary and multiclass cases."""
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        return auc(recall, precision)
    else:
        # Macro-average AUPRC for multiclass (one-vs-rest)
        auprc_scores = []
        y_one_hot = np.eye(n_classes)[y_true]
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_one_hot[:, i], y_prob[:, i])
            auprc_scores.append(auc(recall, precision))
        return np.mean(auprc_scores)

class MLAgentLongitudinalClassifier:
    def __init__(self):
        self.models = None
        self.best_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.ids_val = None
        self.ids_train = None
        self.le_dict = {}
        self.feature_columns = None
        self.n_classes_ = None
        self.label_encoder_ = LabelEncoder()
        self.optimal_cutoffs_ = None  # Store optimal probability cutoffs

    def preprocess_data(self, df, feature_columns, target_column, id_column, time_column):
        required_cols = feature_columns + [id_column, time_column]
        if target_column is not None:
            required_cols.append(target_column)
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        df_processed = df.copy()
        categorical_cols = [col for col in feature_columns if df_processed[col].dtype == 'object']
        if categorical_cols:
            print(f"Encoding categorical columns: {categorical_cols}")
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = df_processed[col].fillna('MISSING')
                df_processed[col] = le.fit_transform(df_processed[col])
                self.le_dict[col] = le

        numeric_cols = [col for col in feature_columns if col not in categorical_cols]
        if df_processed[numeric_cols].isna().any().any():
            print(f"Imputing NaN values in numeric columns: {numeric_cols}")
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

        if target_column is not None:
            if df_processed[target_column].isna().any():
                print(f"Warning: NaNs in target column '{target_column}'. Dropping affected IDs.")
                df_processed = df_processed.groupby(id_column).filter(lambda x: x[target_column].notna().all())
            df_processed[target_column] = self.label_encoder_.fit_transform(df_processed[target_column])
            self.n_classes_ = len(self.label_encoder_.classes_)

        df_sorted = df_processed.sort_values([id_column, time_column])
        unique_ids = df_sorted[id_column].unique()
        n_unique_ids = len(unique_ids)
        X = []
        y = np.zeros(n_unique_ids, dtype=np.int64) if target_column is not None else None
        ids = []

        for i, play_id in enumerate(unique_ids):
            df_id = df_sorted[df_sorted[id_column] == play_id]
            features = df_id[feature_columns].values
            X.append(features)
            if target_column is not None:
                y[i] = df_id[target_column].iloc[-1]
            ids.extend([play_id] * len(features))

        ids = np.array(ids)
        self.feature_columns = feature_columns
        print(f"Processed {n_unique_ids} unique IDs with variable time steps.")
        return X, y, ids

    def compute_optimal_cutoff(self, y_true, y_prob):
        """Compute optimal probability cutoff maximizing sensitivity and specificity."""
        if self.n_classes_ == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
            # Maximize geometric mean of sensitivity (tpr) and specificity (1-fpr)
            gmean = np.sqrt(tpr * (1 - fpr))
            optimal_idx = np.argmax(gmean)
            optimal_threshold = thresholds[optimal_idx]
            return optimal_threshold
        else:
            # For multiclass, compute per-class thresholds (one-vs-rest)
            cutoffs = {}
            y_one_hot = np.eye(self.n_classes_)[y_true]
            for i in range(self.n_classes_):
                fpr, tpr, thresholds = roc_curve(y_one_hot[:, i], y_prob[:, i])
                gmean = np.sqrt(tpr * (1 - fpr))
                optimal_idx = np.argmax(gmean)
                cutoffs[i] = thresholds[optimal_idx]
            return cutoffs

    def plot_roc_prc_curves(self, y_true, y_prob, model_name):
        """Plot ROC and PRC curves for binary or multiclass classification."""
        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        if self.n_classes_ == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'ROC Curve - {model_name} (Binary)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
        else:
            y_one_hot = np.eye(self.n_classes_)[y_true]
            for i in range(self.n_classes_):
                fpr, tpr, _ = roc_curve(y_one_hot[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {self.label_encoder_.classes_[i]} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'ROC Curve - {model_name} (Multiclass)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')

        # PRC Curve
        plt.subplot(1, 2, 2)
        if self.n_classes_ == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            prc_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'PRC curve (AUPRC = {prc_auc:.2f})')
            plt.title(f'Precision-Recall Curve - {model_name} (Binary)')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
        else:
            y_one_hot = np.eye(self.n_classes_)[y_true]
            for i in range(self.n_classes_):
                precision, recall, _ = precision_recall_curve(y_one_hot[:, i], y_prob[:, i])
                prc_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f'Class {self.label_encoder_.classes_[i]} (AUPRC = {prc_auc:.2f})')
            plt.title(f'Precision-Recall Curve - {model_name} (Multiclass)')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')

        plt.tight_layout()
        plt.show()

    def fit(self, X, y, ids=None, batch_size=32, epochs=10):
        if ids is None:
            raise ValueError("ids must be provided for custom train-test split.")
        if not isinstance(X, list):
            raise ValueError("X must be a list of 2D arrays for variable time steps.")
        n_unique_ids = len(y)
        if len(ids) != sum(x.shape[0] for x in X):
            raise ValueError(f"Length of ids ({len(ids)}) must equal total measurements ({sum(x.shape[0] for x in X)})")
    
        unique_ids = np.unique(ids)
        test_size = max(1, int(0.2 * n_unique_ids))
        np.random.seed(42)
        np.random.shuffle(unique_ids)
        test_ids = unique_ids[:test_size]
        train_ids = unique_ids[test_size:]
    
        self.X_train = []
        self.y_train = []
        self.ids_train = []
        self.X_val = []
        self.y_val = []
        self.ids_val = []
    
        start_idx = 0
        for i in range(n_unique_ids):
            n_timesteps = X[i].shape[0]
            id_i = ids[start_idx]
            if id_i in train_ids:
                self.X_train.append(X[i])
                self.y_train.append(y[i])
                self.ids_train.extend([id_i] * n_timesteps)
            elif id_i in test_ids:
                self.X_val.append(X[i])
                self.y_val.append(y[i])
                self.ids_val.extend([id_i] * n_timesteps)
            start_idx += n_timesteps
    
        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)
        self.ids_train = np.array(self.ids_train)
        self.ids_val = np.array(self.ids_val)
    
        if len(self.X_val) == 0 or len(self.y_val) == 0:
            raise ValueError("Validation set is empty. Increase test_size or dataset size.")
    
        y_train_expanded = np.concatenate([np.full(x.shape[0], y_val) for x, y_val in zip(self.X_train, self.y_train)])
        input_dim = X[0].shape[-1]
    
        self.models = {
            'REEM': REEMLogisticClassifier(),
            'RandomForest': LongitudinalRFClassifier(),
            'HMM': HMMClassifier(n_components=2),
            'GPBoost': GPBoostClassifier(),
            'TCN': TCNClassifier(input_dim=input_dim, n_classes=self.n_classes_),
            'LSTM': LSTMClassifier(input_dim=input_dim, n_classes=self.n_classes_),
            'GRU': GRUClassifier(input_dim=input_dim, n_classes=self.n_classes_),
            'Transformer': TransformerClassifier(input_dim=input_dim, n_classes=self.n_classes_)
        }
    
        performances = {}
    
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            if name in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                dataset = TimeSeriesDataset(self.X_train, self.y_train, self.ids_train)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: self._pad_batch(batch))
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                model.to(self.device)
    
                for epoch in range(epochs):
                    model.train()
                    for X_batch, y_batch, _ in loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
    
                model.eval()
                with torch.no_grad():
                    X_val_padded = self._pad_sequences(self.X_val)
                    X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32).to(self.device)
                    y_val_probs = model(X_val_tensor).cpu().numpy()
            else:
                X_train_flat = np.vstack(self.X_train)
                ids_train_flat = np.array(self.ids_train)
                if name == 'REEM':
                    model.fit(X_train_flat, y_train_expanded, ids_train_flat)
                else:
                    model.fit(X_train_flat, y_train_expanded)
                y_val_probs = model.predict_proba(self.X_val) if name != 'REEM' else model.predict_proba(self.X_val, self.ids_val)
    
            auc_roc = auc_roc_score(self.y_val, y_val_probs, self.n_classes_)
            auprc = auprc_score(self.y_val, y_val_probs, self.n_classes_)
            performances[name] = {'auc_roc': auc_roc, 'auprc': auprc}
            print(f"{name} - AUC-ROC: {auc_roc:.4f}, AUPRC: {auprc:.4f}")
    
        # Select best model based on AUC-ROC (primary) and AUPRC (secondary)
        self.best_model = max(performances, key=lambda k: (performances[k]['auc_roc'], performances[k]['auprc']))
        print(f"Best model: {self.best_model} with AUC-ROC: {performances[self.best_model]['auc_roc']:.4f}, AUPRC: {performances[self.best_model]['auprc']:.4f}")
    
        # Compute optimal cutoffs for the best model
        best_model_obj = self.models[self.best_model]
        if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
            best_model_obj.eval()
            with torch.no_grad():
                X_val_padded = self._pad_sequences(self.X_val)
                X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32).to(self.device)
                y_val_probs = best_model_obj(X_val_tensor).cpu().numpy()
        else:
            y_val_probs = best_model_obj.predict_proba(self.X_val) if self.best_model != 'REEM' else best_model_obj.predict_proba(self.X_val, self.ids_val)
        
        self.optimal_cutoffs_ = self.compute_optimal_cutoff(self.y_val, y_val_probs)
        print(f"Optimal cutoffs for {self.best_model}: {self.optimal_cutoffs_}")

    def _pad_batch(self, batch):
        X_batch, y_batch, ids_batch = zip(*batch)
        max_len = max(x.shape[0] for x in X_batch)
        X_padded = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])], dim=0) if x.shape[0] < max_len else x for x in X_batch])
        return X_padded, torch.tensor(y_batch, dtype=torch.long), ids_batch

    def _pad_sequences(self, X):
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

    def predict_proba(self, X, ids=None):
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction.")
        model = self.models[self.best_model]
        if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
            model.eval()
            with torch.no_grad():
                X_padded = self._pad_sequences(X)
                X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(self.device)
                return model(X_tensor).cpu().numpy()
        else:
            return model.predict_proba(X, ids) if self.best_model == 'REEM' else model.predict_proba(X)

    def predict(self, X, ids=None):
        probs = self.predict_proba(X, ids)
        if self.optimal_cutoffs_ is None:
            return np.argmax(probs, axis=1)
        else:
            if self.n_classes_ == 2:
                # Binary: Apply optimal threshold to positive class probability
                return (probs[:, 1] >= self.optimal_cutoffs_).astype(int)
            else:
                # Multiclass: Apply per-class thresholds or fall back to argmax
                preds = np.zeros(len(probs), dtype=int)
                for i in range(len(probs)):
                    for cls, thresh in self.optimal_cutoffs_.items():
                        if probs[i, cls] >= thresh:
                            preds[i] = cls
                            break
                    else:
                        preds[i] = np.argmax(probs[i])
                return preds

    def predict_df(self, df, feature_columns, id_column, time_column):
        X, _, ids = self.preprocess_data(df, feature_columns, target_column=None, id_column=id_column, time_column=time_column)
        preds = self.predict(X, ids)
        return self.label_encoder_.inverse_transform(preds)

    def print_predictions(self, n=10):
        if self.best_model is None or self.X_val is None or self.y_val is None or self.ids_val is None:
            raise ValueError("Fit the model first to generate validation set predictions.")
        val_preds = self.predict(self.X_val, self.ids_val)
        val_preds_labels = self.label_encoder_.inverse_transform(val_preds)
        val_true_labels = self.label_encoder_.inverse_transform(self.y_val)
        print(f"\nValidation Set Predictions (Best Model: {self.best_model}) - First {n} values:")
        print("Predicted\tActual")
        for pred, actual in zip(val_preds_labels[:n], val_true_labels[:n]):
            print(f"{pred}\t\t{actual}")

    def display_target_distribution(self):
        y_train_display = self.label_encoder_.inverse_transform(self.y_train.astype(int))
        y_series = pd.Series(y_train_display)
        total_count = len(y_series)
        
        counts = y_series.value_counts().sort_index()
        percentages = (counts / total_count * 100).round(2)
        
        dist_df = pd.DataFrame({'Class': counts.index, 'Count': counts.values, 'Percentage': percentages.values})
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Class', y='Count', data=dist_df, palette='Blues_d')
        
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.annotate(f'{int(height)}\n({dist_df["Percentage"].iloc[i]}%)',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        
        plt.title("Distribution of Target Outcome in Training Set")
        plt.xlabel("Target Class")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    def compute_prediction_errors(self, X, y, ids, n_bootstraps=1000, random_state=42):
        np.random.seed(random_state)
        y_pred = self.predict(X, ids)
        y_pred_proba = self.predict_proba(X, ids)
        n_samples = len(y)

        def bootstrap_errors(y_true, y_pred_proba):
            auc_roc = auc_roc_score(y_true, y_pred_proba, self.n_classes_)
            auprc = auprc_score(y_true, y_pred_proba, self.n_classes_)
            accuracy = accuracy_score(y_true, np.argmax(y_pred_proba, axis=1))
            return np.array([auc_roc, auprc, accuracy])

        point_estimates = bootstrap_errors(y, y_pred_proba)
        bootstrap_stats = []

        time_steps = [x.shape[0] for x in X]
        start_indices = np.cumsum([0] + time_steps[:-1])

        for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = [X[i] for i in indices]
            y_boot = y[indices]
            ids_boot = np.concatenate([np.full(X[i].shape[0], ids[start_indices[i]]) for i in indices])
            y_pred_boot_proba = self.predict_proba(X_boot, ids_boot)
            stats = bootstrap_errors(y_boot, y_pred_boot_proba)
            bootstrap_stats.append(stats)

        bootstrap_stats = np.array(bootstrap_stats)
        ci_lower = np.percentile(bootstrap_stats, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_stats, 97.5, axis=0)
        metric_names = ["AUC-ROC", "AUPRC", "Accuracy"]
        results = {}
        for i, name in enumerate(metric_names):
            results[name] = {
                "Estimate": point_estimates[i],
                "95% CI": (ci_lower[i], ci_upper[i])
            }

        print(f"\nPrediction Errors for {self.best_model}:")
        for name, res in results.items():
            print(f"{name}: {res['Estimate']:.4f} (95% CI: {res['95% CI'][0]:.4f} - {res['95% CI'][1]:.4f})")
        return results

    def plot_feature_importance(self):
        if self.best_model is None or self.models is None:
            print("No best model has been identified. Train the model first.")
            return

        if not self.feature_columns:
            print("Feature names not provided.")
            return

        def compute_permutation_importance_torch(model, X, y, ids, feature_idx, device, n_repeats=10):
            model.eval()
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
            with torch.no_grad():
                baseline_pred = model(X_tensor).cpu().numpy()
            baseline_auc = auc_roc_score(y, baseline_pred, self.n_classes_)
            scores = []

            for _ in range(n_repeats):
                X_permuted = [x.copy() for x in X]
                for i in range(len(X_permuted)):
                    np.random.shuffle(X_permuted[i][:, feature_idx])
                X_permuted_padded = self._pad_sequences(X_permuted)
                X_permuted_tensor = torch.tensor(X_permuted_padded, dtype=torch.float32).to(device)
                with torch.no_grad():
                    permuted_pred = model(X_permuted_tensor).cpu().numpy()
                permuted_auc = auc_roc_score(y, permuted_pred, self.n_classes_)
                scores.append(baseline_auc - permuted_auc)

            return np.mean(np.abs(scores))

        model = self.models[self.best_model]
        feature_names = self.feature_columns
        n_features = len(feature_names)

        if self.best_model == 'REEM' and hasattr(model.fixed_model_, 'coef_'):
            importances = np.abs(model.fixed_model_.coef_).mean(axis=0)
            title = f"{self.best_model} (Absolute Coefficients)"
        elif self.best_model == 'RandomForest' and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            title = f"{self.best_model} (Feature Importances)"
        elif self.best_model == 'GPBoost' and hasattr(model.model, 'booster_'):
            booster = model.model.booster_
            importances = booster.feature_importance(importance_type='gain')
            title = f"{self.best_model} (Gain Importance)"
        else:
            importances = np.zeros(n_features)
            if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                for i in range(n_features):
                    importances[i] = compute_permutation_importance_torch(
                        model, self.X_train, self.y_train, self.ids_train, i, self.device
                    )
            else:
                X_train_flat = np.vstack(self.X_train)
                y_train_expanded = np.concatenate([np.full(x.shape[0], y_val) for x, y_val in zip(self.X_train, self.y_train)])
                r = permutation_importance(
                    model, X_train_flat, y_train_expanded,
                    scoring='roc_auc' if self.n_classes_ == 2 else 'roc_auc_ovr',
                    n_repeats=10, random_state=42
                )
                importances = r.importances_mean
            title = f"{self.best_model} (Permutation Importance)"

        importances = importances / np.max(np.abs(importances) + 1e-10)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title(title)
        plt.xlabel("Normalized Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(self, X, y, ids=None):
        y_pred = self.predict(X, ids)
        y_pred_labels = self.label_encoder_.inverse_transform(y_pred)
        y_true_labels = self.label_encoder_.inverse_transform(y)
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels))

        # Plot ROC and PRC for validation set
        if self.X_val is not None and self.y_val is not None:
            y_val_probs = self.predict_proba(self.X_val, self.ids_val)
            self.plot_roc_prc_curves(self.y_val, y_val_probs, self.best_model)

    def train_and_evaluate(self, df_or_X=None, feature_columns=None, target_column=None, id_column=None, time_column=None, X=None, y=None, ids=None, batch_size=32, epochs=10, filepath="best_classifier_model.pkl"):
        if isinstance(df_or_X, pd.DataFrame):
            if any(arg is None for arg in [feature_columns, target_column, id_column, time_column]):
                raise ValueError("feature_columns, target_column, id_column, and time_column must be provided when passing a DataFrame.")
            X, y, ids = self.preprocess_data(df_or_X, feature_columns, target_column, id_column, time_column)
        else:
            if X is None or y is None or ids is None:
                raise ValueError("X, y, and ids must be provided when not passing a DataFrame.")
            X, y, ids = X, y, ids

        print("=== Longitudinal Classification ===")
        print("\nFitting model...")
        self.fit(X, y, ids, batch_size=batch_size, epochs=epochs)

        print("\nDisplaying outcome distribution...")
        self.display_target_distribution()

        print("\nGenerating predictions on full dataset...")
        preds = self.predict(X, ids)

        print("\nComputing prediction errors...")
        self.compute_prediction_errors(X, y, ids)

        print("\nPlotting feature importance...")
        self.plot_feature_importance()

        print("\nPlotting predictions vs actual (including ROC/PRC curves)...")
        self.plot_predictions_vs_actual(X, y, ids)

        print("\nSaving best model...")
        model = self.models[self.best_model]
        try:
            if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                torch.save(model.state_dict(), filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            print(f"Best model ({self.best_model}) saved to {filepath}")
        except Exception as e:
            print(f"Failed to save model due to: {e}")
            print(f"Best model ({self.best_model}) not saved, but training completed successfully.")

        print("\nTraining and evaluation completed.")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from hmmlearn.hmm import GaussianHMM
import gpboost as gpb
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

# Suppress KMeans warning
os.environ["OMP_NUM_THREADS"] = "2"

class REEMRegressor:
    def __init__(self, max_iter=100, tol=1e-4, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_effects_ = None
        self.fixed_model_ = None
        self.scaler_ = StandardScaler()

    def _initialize_random_effects(self, ids):
        return defaultdict(float)

    def _e_step(self, fixed_effects_pred, y, ids):
        new_random_effects = defaultdict(float)
        for i in np.unique(ids):
            idx = (ids == i)
            y_i = y[idx]
            fixed_pred_i = fixed_effects_pred[idx]

            def loss(bi):
                return np.mean((y_i - (fixed_pred_i + bi)) ** 2)

            best_bi, best_loss = 0, np.inf
            for bi in np.linspace(-3, 3, 21):
                l = loss(bi)
                if l < best_loss:
                    best_loss, best_bi = l, bi
            new_random_effects[i] = best_bi
        return new_random_effects

    def _m_step(self, X, y, random_effects, ids):
        offset = np.array([random_effects.get(i, 0.0) for i in ids])
        adjusted_y = y - offset
        model = LinearRegression()
        model.fit(X, adjusted_y)
        return model

    def fit(self, X, y, ids):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            ids_flat = np.concatenate([np.full(x.shape[0], id_val) for x, id_val in zip(X, ids)])
            if len(ids) != len(X):
                raise ValueError(f"Length of ids ({len(ids)}) must match number of sequences ({len(X)})")
        else:
            X_flat = X
            ids_flat = ids
            if X_flat.shape[0] != len(ids_flat):
                raise ValueError(f"X rows ({X_flat.shape[0]}) must match length of ids ({len(ids_flat)})")

        if X_flat.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows: got {X_flat.shape[0]} and {y.shape[0]}")

        if np.any(np.isnan(X_flat)) or np.any(np.isnan(y)):
            raise ValueError("Input X or y contains NaN values")
        if np.any(np.isinf(X_flat)) or np.any(np.isinf(y)):
            raise ValueError("Input X or y contains infinite values")

        X_scaled = self.scaler_.fit_transform(X_flat)
        self.random_effects_ = self._initialize_random_effects(ids_flat)

        for iteration in tqdm(range(self.max_iter), desc="REEM Fitting", disable=not self.verbose):
            self.fixed_model_ = self._m_step(X_scaled, y, self.random_effects_, ids_flat)
            fixed_preds = self.fixed_model_.predict(X_scaled)
            new_random_effects = self._e_step(fixed_preds, y, ids_flat)

            diffs = np.array([
                abs(self.random_effects_[i] - new_random_effects[i])
                for i in new_random_effects if i in self.random_effects_
            ])
            self.random_effects_ = new_random_effects
            if len(diffs) > 0 and np.max(diffs) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            elif len(diffs) == 0:
                if self.verbose:
                    print("No overlapping IDs found for convergence check")
                break

        return self

    def predict(self, X, ids):
        if isinstance(X, list):
            n_samples = len(X)
            X_flat = np.vstack(X)
            full_ids = ids
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
            if len(full_ids) != sum(n_timesteps_per_sample):
                raise ValueError(f"Length of ids ({len(full_ids)}) must match total time steps ({sum(n_timesteps_per_sample)})")
        else:
            X_flat = X
            full_ids = ids
            n_timesteps_per_sample = [1] * X.shape[0]

        X_flat = self.scaler_.transform(X_flat)
        preds_flat = self.fixed_model_.predict(X_flat)
        offset_flat = np.array([self.random_effects_.get(i, 0.0) for i in full_ids])
        preds_with_offset = preds_flat + offset_flat

        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_with_offset[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y, ids):
        preds = self.predict(X, ids)
        return -mean_squared_error(y, preds)

class LongitudinalRFRegressor:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        preds_flat = self.model.predict(X_scaled)
        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_flat[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return -mean_squared_error(y, preds)

class HMMRegressor:
    def __init__(self, n_components=2):
        self.model = GaussianHMM(n_components=n_components)
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.regressor = LinearRegression()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        states = self.model.predict(X_scaled)
        self.regressor.fit(states.reshape(-1, 1), y)

    def predict(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        states = self.model.predict(X_scaled)
        preds_flat = self.regressor.predict(states.reshape(-1, 1))
        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_flat[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return -mean_squared_error(y, preds)

class GPBoostRegressor:
    def __init__(self):
        self.model = gpb.GPBoostRegressor()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        preds_flat = self.model.predict(X_scaled)
        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_flat[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return -mean_squared_error(y, preds)

class TCNRegressor(nn.Module):
    def __init__(self, input_dim, num_channels=[16, 32], kernel_size=2, dropout=0.2):
        super(TCNRegressor, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          dilation=dilation_size, padding=(kernel_size-1)*dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        return self.fc(x)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.fc(hn[-1])

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids[idx]

class MLAgentLongitudinalRegressor:
    def __init__(self):
        self.models = None
        self.best_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.ids_val = None
        self.ids_train = None
        self.le_dict = {}
        self.feature_columns = None

    def preprocess_data(self, df, feature_columns, target_column, id_column, time_column):
        missing_cols = [col for col in feature_columns + ([target_column] if target_column else []) + [id_column, time_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        df_processed = df.copy()
        categorical_cols = [col for col in feature_columns if df_processed[col].dtype == 'object']
        if categorical_cols:
            print(f"Encoding categorical columns: {categorical_cols}")
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = df_processed[col].fillna('MISSING')
                df_processed[col] = le.fit_transform(df_processed[col])
                self.le_dict[col] = le

        numeric_cols = [col for col in feature_columns if col not in categorical_cols]
        if df_processed[numeric_cols].isna().any().any():
            print(f"Imputing NaN values in numeric columns: {numeric_cols}")
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

        if target_column and df_processed[target_column].isna().any():
            print(f"Warning: NaNs in target column '{target_column}'. Dropping affected IDs.")
            df_processed = df_processed.groupby(id_column).filter(lambda x: x[target_column].notna().all())

        df_sorted = df_processed.sort_values([id_column, time_column])
        unique_ids = df_sorted[id_column].unique()
        n_unique_ids = len(unique_ids)
        X = []
        y = np.zeros(n_unique_ids) if target_column else None
        ids = []

        for i, play_id in enumerate(unique_ids):
            df_id = df_sorted[df_sorted[id_column] == play_id]
            features = df_id[feature_columns].values
            X.append(features)
            if target_column:
                y[i] = df_id[target_column].iloc[-1]
            ids.extend([play_id] * len(features))

        ids = np.array(ids)
        self.feature_columns = feature_columns
        print(f"Processed {n_unique_ids} unique IDs with variable time steps.")
        return X, y, ids

    def fit(self, X, y, ids=None, batch_size=32, epochs=10):
        if ids is None:
            raise ValueError("ids must be provided for custom train-test split.")
        if not isinstance(X, list):
            raise ValueError("X must be a list of 2D arrays for variable time steps.")
        n_unique_ids = len(y)
        if len(ids) != sum(x.shape[0] for x in X):
            raise ValueError(f"Length of ids ({len(ids)}) must equal total measurements ({sum(x.shape[0] for x in X)})")

        unique_ids = np.unique(ids)
        test_size = max(1, int(0.2 * n_unique_ids))
        np.random.seed(42)
        np.random.shuffle(unique_ids)
        test_ids = unique_ids[:test_size]
        train_ids = unique_ids[test_size:]

        self.X_train = []
        self.y_train = []
        self.ids_train = []
        self.X_val = []
        self.y_val = []
        self.ids_val = []

        start_idx = 0
        for i in range(n_unique_ids):
            n_timesteps = X[i].shape[0]
            id_i = ids[start_idx]
            if id_i in train_ids:
                self.X_train.append(X[i])
                self.y_train.append(y[i])
                self.ids_train.extend([id_i] * n_timesteps)
            elif id_i in test_ids:
                self.X_val.append(X[i])
                self.y_val.append(y[i])
                self.ids_val.extend([id_i] * n_timesteps)
            start_idx += n_timesteps

        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)
        self.ids_train = np.array(self.ids_train)
        self.ids_val = np.array(self.ids_val)

        if len(self.X_val) == 0 or len(self.y_val) == 0:
            raise ValueError("Validation set is empty. Increase test_size or dataset size.")

        y_train_expanded = np.concatenate([np.full(x.shape[0], y_val) for x, y_val in zip(self.X_train, self.y_train)])
        input_dim = X[0].shape[-1]

        self.models = {
            'REEM': REEMRegressor(),
            'RandomForest': LongitudinalRFRegressor(),
            'HMM': HMMRegressor(n_components=2),
            'GPBoost': GPBoostRegressor(),
            'TCN': TCNRegressor(input_dim=input_dim, num_channels=[16, 32]),
            'LSTM': LSTMRegressor(input_dim=input_dim, hidden_dim=64),
            'GRU': GRURegressor(input_dim=input_dim, hidden_dim=64),
            'Transformer': TransformerRegressor(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
        }

        performances = {}

        for name, model in self.models.items():
            print(f"Fitting {name}...")
            if name in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                dataset = TimeSeriesDataset(self.X_train, self.y_train, self.ids_train)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: self._pad_batch(batch))
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                model.to(self.device)

                for epoch in range(epochs):
                    model.train()
                    for X_batch, y_batch, _ in loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).view(-1, 1)
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    X_val_padded = self._pad_sequences(self.X_val)
                    X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32).to(self.device)
                    y_val_pred = model(X_val_tensor).cpu().numpy().flatten()
            else:
                X_train_flat = np.vstack(self.X_train)
                ids_train_flat = np.array(self.ids_train)
                if name == 'REEM':
                    model.fit(X_train_flat, y_train_expanded, ids_train_flat)
                else:
                    model.fit(X_train_flat, y_train_expanded)
                y_val_pred = model.predict(self.X_val) if name != 'REEM' else model.predict(self.X_val, self.ids_val)

            mse = mean_squared_error(self.y_val, y_val_pred)
            mae = mean_absolute_error(self.y_val, y_val_pred)
            r2 = r2_score(self.y_val, y_val_pred)
            performances[name] = {'mse': mse, 'r2': r2}
            print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        self.best_model = min(performances, key=lambda k: performances[k]['mse'])
        print(f"Best model: {self.best_model} with MSE: {performances[self.best_model]['mse']:.4f}, R²: {performances[self.best_model]['r2']:.4f}")

    def _pad_batch(self, batch):
        X_batch, y_batch, ids_batch = zip(*batch)
        max_len = max(x.shape[0] for x in X_batch)
        X_padded = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])], dim=0) if x.shape[0] < max_len else x for x in X_batch])
        return X_padded, torch.stack(y_batch), ids_batch

    def _pad_sequences(self, X):
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

    def predict(self, X, ids=None):
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction.")
        model = self.models[self.best_model]
        if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
            model.eval()
            with torch.no_grad():
                X_padded = self._pad_sequences(X)
                X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(self.device)
                return model(X_tensor).cpu().numpy().flatten()
        else:
            return model.predict(X, ids) if self.best_model == 'REEM' else model.predict(X)

    def predict_df(self, df, feature_columns, id_column, time_column):
        X, _, ids = self.preprocess_data(df, feature_columns, target_column=None, id_column=id_column, time_column=time_column)
        return self.predict(X, ids)

    def print_predictions(self, n=10):
        if self.best_model is None or self.X_val is None or self.y_val is None or self.ids_val is None:
            raise ValueError("Fit the model first to generate validation set predictions.")
        val_preds = self.predict(self.X_val, self.ids_val)
        print(f"\nValidation Set Predictions (Best Model: {self.best_model}) - First {n} values:")
        print("Predicted\tActual")
        for pred, actual in zip(val_preds[:n], self.y_val[:n]):
            print(f"{pred:.4f}\t\t{actual:.4f}")

    def display_outcome_distribution(self, y):
        plt.figure(figsize=(8, 6))
        sns.histplot(y, bins=30, kde=True)
        plt.title("Outcome Distribution (Regression)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    def compute_prediction_errors(self, X, y, ids, n_bootstraps=1000, random_state=42):
        np.random.seed(random_state)
        y_pred = self.predict(X, ids)
        n_samples = len(y)

        def bootstrap_errors(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return np.array([mse, mae, r2])

        point_estimates = bootstrap_errors(y, y_pred)
        bootstrap_stats = []

        time_steps = [x.shape[0] for x in X]
        start_indices = np.cumsum([0] + time_steps[:-1])

        for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = [X[i] for i in indices]
            y_boot = y[indices]
            ids_boot = np.concatenate([np.full(X[i].shape[0], ids[start_indices[i]]) for i in indices])
            y_pred_boot = self.predict(X_boot, ids_boot)
            stats = bootstrap_errors(y_boot, y_pred_boot)
            bootstrap_stats.append(stats)

        bootstrap_stats = np.array(bootstrap_stats)
        ci_lower = np.percentile(bootstrap_stats, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_stats, 97.5, axis=0)
        metric_names = ["MSE", "MAE", "R²"]
        results = {}
        for i, name in enumerate(metric_names):
            results[name] = {
                "Estimate": point_estimates[i],
                "95% CI": (ci_lower[i], ci_upper[i])
            }

        print(f"\nPrediction Errors for {self.best_model}:")
        for name, res in results.items():
            print(f"{name}: {res['Estimate']:.4f} (95% CI: {res['95% CI'][0]:.4f} - {res['95% CI'][1]:.4f})")
        return results

    def plot_feature_importance(self):
        if self.best_model is None or self.models is None:
            print("No best model has been identified. Train the model first.")
            return
    
        if not self.feature_columns:
            print("Feature names not provided.")
            return
    
        def compute_permutation_importance_torch(model, X, y, ids, feature_idx, device, n_repeats=10):
            model.eval()
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
            with torch.no_grad():
                baseline_pred = model(X_tensor).cpu().numpy().flatten()
            baseline_mse = mean_squared_error(y, baseline_pred)
            scores = []
    
            for _ in range(n_repeats):
                X_permuted = [x.copy() for x in X]
                for i in range(len(X_permuted)):
                    np.random.shuffle(X_permuted[i][:, feature_idx])
                X_permuted_padded = self._pad_sequences(X_permuted)
                X_permuted_tensor = torch.tensor(X_permuted_padded, dtype=torch.float32).to(device)
                with torch.no_grad():
                    permuted_pred = model(X_permuted_tensor).cpu().numpy().flatten()
                permuted_mse = mean_squared_error(y, permuted_pred)
                scores.append(permuted_mse - baseline_mse)
    
            return np.mean(scores)
    
        model = self.models[self.best_model]
        feature_names = self.feature_columns
        n_features = len(feature_names)
        
        if self.best_model == 'REEM' and hasattr(model.fixed_model_, 'coef_'):
            importances = np.abs(model.fixed_model_.coef_)
            title = f"{self.best_model} (Absolute Coefficients)"
        elif self.best_model == 'RandomForest' and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            title = f"{self.best_model} (Feature Importances)"
        elif self.best_model == 'GPBoost' and hasattr(model.model, 'booster_'):
            booster = model.model.booster_
            importances = booster.feature_importance(importance_type='gain')
            if len(importances) != n_features:
                raise ValueError(f"GPBoost importance length ({len(importances)}) does not match feature count ({n_features})")
            title = f"{self.best_model} (Gain Importance)"
        else:
            importances = np.zeros(n_features)
            if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                for i in range(n_features):
                    importances[i] = compute_permutation_importance_torch(
                        model, self.X_train, self.y_train, self.ids_train, i, self.device
                    )
            else:
                X_train_flat = np.vstack(self.X_train)
                y_train_expanded = np.concatenate([np.full(x.shape[0], y_val) for x, y_val in zip(self.X_train, self.y_train)])
                r = permutation_importance(
                    model, X_train_flat, y_train_expanded, 
                    scoring='neg_mean_squared_error', n_repeats=10, random_state=42
                )
                importances = r.importances_mean
            title = f"{self.best_model} (Permutation Importance)"
    
        importances = importances / np.max(np.abs(importances) + 1e-10)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title(title)
        plt.xlabel("Normalized Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(self, X, y, ids=None):
        y_pred = self.predict(X, ids)
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title(f"Predictions vs Actual ({self.best_model})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.show()

    def train_and_evaluate(self, df_or_X=None, feature_columns=None, target_column=None, id_column=None, time_column=None, X=None, y=None, ids=None, batch_size=32, epochs=10, filepath="best_model.pkl"):
        if isinstance(df_or_X, pd.DataFrame):
            if any(arg is None for arg in [feature_columns, target_column, id_column, time_column]):
                raise ValueError("feature_columns, target_column, id_column, and time_column must be provided when passing a DataFrame.")
            X, y, ids = self.preprocess_data(df_or_X, feature_columns, target_column, id_column, time_column)
        else:
            if X is None or y is None or ids is None:
                raise ValueError("X, y, and ids must be provided when not passing a DataFrame.")
            X, y, ids = X, y, ids

        print("=== Longitudinal Regression ===")
        print("\nFitting model...")
        self.fit(X, y, ids, batch_size=batch_size, epochs=epochs)
        
        print("\nDisplaying outcome distribution...")
        self.display_outcome_distribution(y)
        
        print("\nGenerating predictions on full dataset...")
        preds = self.predict(X, ids)
        
        print("\nComputing prediction errors...")
        self.compute_prediction_errors(X, y, ids)
        
        print("\nPlotting feature importance for all models...")
        self.plot_feature_importance()
        
        print("\nPlotting predictions vs actual...")
        self.plot_predictions_vs_actual(X, y, ids)
        
        print("\nSaving best model...")
        model = self.models[self.best_model]
        if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
            torch.save(model.state_dict(), filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        print(f"Best model ({self.best_model}) saved to {filepath}")
           
        print("\nTraining and evaluation completed.")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from hmmlearn.hmm import GaussianHMM
import gpboost as gpb
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
import warnings

# Suppress warnings
os.environ["OMP_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

class REEMPoissonRegressor:
    def __init__(self, max_iter=100, tol=1e-4, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_effects_ = None
        self.fixed_model_ = None
        self.scaler_ = StandardScaler()

    def _initialize_random_effects(self, ids):
        return defaultdict(float)

    def _e_step(self, fixed_effects_pred, y, ids):
        new_random_effects = defaultdict(float)
        for i in np.unique(ids):
            idx = (ids == i)
            y_i = y[idx]
            fixed_pred_i = fixed_effects_pred[idx]

            def poisson_loss(bi):
                rate = np.exp(fixed_pred_i + bi)
                return -np.mean(y_i * np.log(rate + 1e-10) - rate)

            best_bi, best_loss = 0, np.inf
            for bi in np.linspace(-3, 3, 21):
                l = poisson_loss(bi)
                if l < best_loss:
                    best_loss, best_bi = l, bi
            new_random_effects[i] = best_bi
        return new_random_effects

    def _m_step(self, X, y, random_effects, ids):
        offset = np.array([random_effects.get(i, 0.0) for i in ids])
        model = PoissonRegressor(max_iter=1000)
        model.fit(X, y, sample_weight=np.exp(-offset))
        return model

    def fit(self, X, y, ids):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            ids_flat = np.concatenate([np.full(x.shape[0], id_val) for x, id_val in zip(X, ids)])
            if len(ids) != len(X):
                raise ValueError(f"Length of ids ({len(ids)}) must match number of sequences ({len(X)})")
        else:
            X_flat = X
            ids_flat = ids
            if X_flat.shape[0] != len(ids_flat):
                raise ValueError(f"X rows ({X_flat.shape[0]}) must match length of ids ({len(ids_flat)})")

        if X_flat.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows: got {X_flat.shape[0]} and {y.shape[0]}")
        if np.any(np.isnan(X_flat)) or np.any(np.isnan(y)):
            raise ValueError("Input X or y contains NaN values")
        if np.any(np.isinf(X_flat)) or np.any(np.isinf(y)):
            raise ValueError("Input X or y contains infinite values")
        if np.any(y < 0):
            raise ValueError("Count data must be non-negative")

        X_scaled = self.scaler_.fit_transform(X_flat)
        self.random_effects_ = self._initialize_random_effects(ids_flat)

        for iteration in tqdm(range(self.max_iter), desc="REEM Poisson Fitting", disable=not self.verbose):
            self.fixed_model_ = self._m_step(X_scaled, y, self.random_effects_, ids_flat)
            fixed_preds = self.fixed_model_.predict(X_scaled)
            new_random_effects = self._e_step(fixed_preds, y, ids_flat)

            diffs = np.array([
                abs(self.random_effects_[i] - new_random_effects[i])
                for i in new_random_effects if i in self.random_effects_
            ])
            self.random_effects_ = new_random_effects
            if len(diffs) > 0 and np.max(diffs) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            elif len(diffs) == 0:
                if self.verbose:
                    print("No overlapping IDs found for convergence check")
                break

        return self

    def predict(self, X, ids):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]

        X_flat = self.scaler_.transform(X_flat)
        preds_flat = self.fixed_model_.predict(X_flat)
        offset_flat = np.array([self.random_effects_.get(i, 0.0) for i in ids])
        preds_with_offset = np.exp(np.log(preds_flat + 1e-10) + offset_flat)

        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_with_offset[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y, ids):
        preds = self.predict(X, ids)
        return -self._poisson_deviance(y, preds)

    def _poisson_deviance(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, np.inf)
        dev = 2 * (y_true * np.log(y_true / y_pred + 1e-10) - (y_true - y_pred))
        return np.mean(dev)

class LongitudinalPoissonRFRegressor:
    def __init__(self):
        self.model = RandomForestRegressor()  # Note: RF not ideal for Poisson, used as baseline
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        preds_flat = np.clip(self.model.predict(X_scaled), 0, np.inf)
        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_flat[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return -self._poisson_deviance(y, preds)

    def _poisson_deviance(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, np.inf)
        dev = 2 * (y_true * np.log(y_true / y_pred + 1e-10) - (y_true - y_pred))
        return np.mean(dev)

class HMMCountRegressor:
    def __init__(self, n_components=2):
        self.model = GaussianHMM(n_components=n_components)
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.regressor = PoissonRegressor(max_iter=1000)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        states = self.model.predict(X_scaled)
        self.regressor.fit(states.reshape(-1, 1), y)

    def predict(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        states = self.model.predict(X_scaled)
        preds_flat = np.clip(self.regressor.predict(states.reshape(-1, 1)), 0, np.inf)
        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_flat[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return -self._poisson_deviance(y, preds)

    def _poisson_deviance(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, np.inf)
        dev = 2 * (y_true * np.log(y_true / y_pred + 1e-10) - (y_true - y_pred))
        return np.mean(dev)

class GPBoostCountRegressor:
    def __init__(self):
        self.model = gpb.GPBoostRegressor(objective="poisson")
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        if isinstance(X, list):
            X_flat = np.vstack(X)
            n_timesteps_per_sample = [seq.shape[0] for seq in X]
        else:
            X_flat = X
            n_timesteps_per_sample = [1] * X.shape[0]
        X_scaled = self.scaler.transform(X_flat)
        preds_flat = np.clip(self.model.predict(X_scaled), 0, np.inf)
        preds = []
        start = 0
        for n_timesteps in n_timesteps_per_sample:
            end = start + n_timesteps
            preds.append(preds_flat[start:end][-1])
            start = end
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return -self._poisson_deviance(y, preds)

    def _poisson_deviance(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, np.inf)
        dev = 2 * (y_true * np.log(y_true / y_pred + 1e-10) - (y_true - y_pred))
        return np.mean(dev)

class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1e-10)
        loss = y_pred - y_true * torch.log(y_pred)
        return torch.mean(loss)

class TCNCountRegressor(nn.Module):
    def __init__(self, input_dim, num_channels=[16, 32], kernel_size=2, dropout=0.2):
        super(TCNCountRegressor, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          dilation=dilation_size, padding=(kernel_size-1)*dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        return torch.exp(self.fc(x))  # Ensure positive outputs for counts

class LSTMCountRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMCountRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return torch.exp(self.fc(hn[-1]))  # Ensure positive outputs

class GRUCountRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GRUCountRegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return torch.exp(self.fc(hn[-1]))  # Ensure positive outputs

class TransformerCountRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerCountRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return torch.exp(self.fc(x[:, -1, :]))  # Ensure positive outputs

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids[idx]

class MLAgentLongitudinalCountRegressor:
    def __init__(self):
        self.models = None
        self.best_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.ids_val = None
        self.ids_train = None
        self.le_dict = {}
        self.feature_columns = None

    def preprocess_data(self, df, feature_columns, target_column, id_column, time_column):
        missing_cols = [col for col in feature_columns + ([target_column] if target_column else []) + [id_column, time_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        df_processed = df.copy()
        categorical_cols = [col for col in feature_columns if df_processed[col].dtype == 'object']
        if categorical_cols:
            print(f"Encoding categorical columns: {categorical_cols}")
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = df_processed[col].fillna('MISSING')
                df_processed[col] = le.fit_transform(df_processed[col])
                self.le_dict[col] = le

        numeric_cols = [col for col in feature_columns if col not in categorical_cols]
        if df_processed[numeric_cols].isna().any().any():
            print(f"Imputing NaN values in numeric columns: {numeric_cols}")
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

        if target_column:
            if df_processed[target_column].isna().any():
                print(f"Warning: NaNs in target column '{target_column}'. Dropping affected IDs.")
                df_processed = df_processed.groupby(id_column).filter(lambda x: x[target_column].notna().all())
            if (df_processed[target_column] < 0).any():
                raise ValueError("Target column contains negative values, which are invalid for count data")

        df_sorted = df_processed.sort_values([id_column, time_column])
        unique_ids = df_sorted[id_column].unique()
        n_unique_ids = len(unique_ids)
        X = []
        y = np.zeros(n_unique_ids, dtype=np.int64) if target_column else None
        ids = []

        for i, play_id in enumerate(unique_ids):
            df_id = df_sorted[df_sorted[id_column] == play_id]
            features = df_id[feature_columns].values
            X.append(features)
            if target_column:
                y[i] = df_id[target_column].iloc[-1]
            ids.extend([play_id] * len(features))

        ids = np.array(ids)
        self.feature_columns = feature_columns
        print(f"Processed {n_unique_ids} unique IDs with variable time steps.")
        return X, y, ids

    def fit(self, X, y, ids=None, batch_size=32, epochs=10):
        if ids is None:
            raise ValueError("ids must be provided for custom train-test split.")
        if not isinstance(X, list):
            raise ValueError("X must be a list of 2D arrays for variable time steps.")
        n_unique_ids = len(y)
        if len(ids) != sum(x.shape[0] for x in X):
            raise ValueError(f"Length of ids ({len(ids)}) must equal total measurements ({sum(x.shape[0] for x in X)})")
        if np.any(y < 0):
            raise ValueError("Count data must be non-negative")

        unique_ids = np.unique(ids)
        test_size = max(1, int(0.2 * n_unique_ids))
        np.random.seed(42)
        np.random.shuffle(unique_ids)
        test_ids = unique_ids[:test_size]
        train_ids = unique_ids[test_size:]

        self.X_train = []
        self.y_train = []
        self.ids_train = []
        self.X_val = []
        self.y_val = []
        self.ids_val = []

        start_idx = 0
        for i in range(n_unique_ids):
            n_timesteps = X[i].shape[0]
            id_i = ids[start_idx]
            if id_i in train_ids:
                self.X_train.append(X[i])
                self.y_train.append(y[i])
                self.ids_train.extend([id_i] * n_timesteps)
            elif id_i in test_ids:
                self.X_val.append(X[i])
                self.y_val.append(y[i])
                self.ids_val.extend([id_i] * n_timesteps)
            start_idx += n_timesteps

        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)
        self.ids_train = np.array(self.ids_train)
        self.ids_val = np.array(self.ids_val)

        if len(self.X_val) == 0 or len(self.y_val) == 0:
            raise ValueError("Validation set is empty. Increase test_size or dataset size.")

        y_train_expanded = np.concatenate([np.full(x.shape[0], y_val) for x, y_val in zip(self.X_train, self.y_train)])
        input_dim = X[0].shape[-1]

        self.models = {
            'REEM': REEMPoissonRegressor(),
            'RandomForest': LongitudinalPoissonRFRegressor(),
            'HMM': HMMCountRegressor(n_components=2),
            'GPBoost': GPBoostCountRegressor(),
            'TCN': TCNCountRegressor(input_dim=input_dim, num_channels=[16, 32]),
            'LSTM': LSTMCountRegressor(input_dim=input_dim, hidden_dim=64),
            'GRU': GRUCountRegressor(input_dim=input_dim, hidden_dim=64),
            'Transformer': TransformerCountRegressor(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
        }

        performances = {}

        for name, model in self.models.items():
            print(f"Fitting {name}...")
            if name in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                dataset = TimeSeriesDataset(self.X_train, self.y_train, self.ids_train)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: self._pad_batch(batch))
                criterion = PoissonLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                model.to(self.device)

                for epoch in range(epochs):
                    model.train()
                    for X_batch, y_batch, _ in loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).view(-1, 1)
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    X_val_padded = self._pad_sequences(self.X_val)
                    X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32).to(self.device)
                    y_val_pred = model(X_val_tensor).cpu().numpy().flatten()
            else:
                X_train_flat = np.vstack(self.X_train)
                ids_train_flat = np.array(self.ids_train)
                if name == 'REEM':
                    model.fit(X_train_flat, y_train_expanded, ids_train_flat)
                else:
                    model.fit(X_train_flat, y_train_expanded)
                y_val_pred = model.predict(self.X_val) if name != 'REEM' else model.predict(self.X_val, self.ids_val)

            # Calculate metrics
            deviance = self._poisson_deviance(self.y_val, y_val_pred)
            mse = mean_squared_error(self.y_val, y_val_pred)
            mae = mean_absolute_error(self.y_val, y_val_pred)
            r2 = r2_score(self.y_val, y_val_pred)

            performances[name] = {
                'deviance': deviance,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            print(f"{name} - Deviance: {deviance:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Select best model based on lowest Poisson deviance
        self.best_model = min(performances, key=lambda k: performances[k]['deviance'])
        print(f"Best model: {self.best_model} with Deviance: {performances[self.best_model]['deviance']:.4f}, "
              f"MSE: {performances[self.best_model]['mse']:.4f}, "
              f"MAE: {performances[self.best_model]['mae']:.4f}, "
              f"R²: {performances[self.best_model]['r2']:.4f}")

    def _poisson_deviance(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, np.inf)
        dev = 2 * (y_true * np.log(y_true / y_pred + 1e-10) - (y_true - y_pred))
        return np.mean(dev)

    def _pad_batch(self, batch):
        X_batch, y_batch, ids_batch = zip(*batch)
        max_len = max(x.shape[0] for x in X_batch)
        X_padded = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])], dim=0) if x.shape[0] < max_len else x for x in X_batch])
        return X_padded, torch.stack(y_batch), ids_batch

    def _pad_sequences(self, X):
        max_len = max(x.shape[0] for x in X)
        return np.stack([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) if x.shape[0] < max_len else x for x in X])

    def predict(self, X, ids=None):
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction.")
        model = self.models[self.best_model]
        if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
            model.eval()
            with torch.no_grad():
                X_padded = self._pad_sequences(X)
                X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(self.device)
                return np.clip(model(X_tensor).cpu().numpy().flatten(), 0, np.inf)
        else:
            return model.predict(X, ids) if self.best_model == 'REEM' else model.predict(X)

    def predict_df(self, df, feature_columns, id_column, time_column):
        X, _, ids = self.preprocess_data(df, feature_columns, target_column=None, id_column=id_column, time_column=time_column)
        return self.predict(X, ids)

    def print_predictions(self, n=10):
        if self.best_model is None or self.X_val is None or self.y_val is None or self.ids_val is None:
            raise ValueError("Fit the model first to generate validation set predictions.")
        val_preds = self.predict(self.X_val, self.ids_val)
        print(f"\nValidation Set Predictions (Best Model: {self.best_model}) - First {n} values:")
        print("Predicted\tActual")
        for pred, actual in zip(val_preds[:n], self.y_val[:n]):
            print(f"{pred:.4f}\t\t{actual:.4f}")

    def display_outcome_distribution(self, y):
        plt.figure(figsize=(8, 6))
        sns.histplot(y, bins=30, kde=False)
        plt.title("Outcome Distribution (Count Data)")
        plt.xlabel("Count")
        plt.ylabel("Frequency")
        plt.show()

    def compute_prediction_errors(self, X, y, ids, n_bootstraps=1000, random_state=42):
        np.random.seed(random_state)
        y_pred = self.predict(X, ids)
        n_samples = len(y)

        def bootstrap_errors(y_true, y_pred):
            deviance = self._poisson_deviance(y_true, y_pred)
            return np.array([deviance])

        point_estimates = bootstrap_errors(y, y_pred)
        bootstrap_stats = []

        time_steps = [x.shape[0] for x in X]
        start_indices = np.cumsum([0] + time_steps[:-1])

        for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = [X[i] for i in indices]
            y_boot = y[indices]
            ids_boot = np.concatenate([np.full(X[i].shape[0], ids[start_indices[i]]) for i in indices])
            y_pred_boot = self.predict(X_boot, ids_boot)
            stats = bootstrap_errors(y_boot, y_pred_boot)
            bootstrap_stats.append(stats)

        bootstrap_stats = np.array(bootstrap_stats)
        ci_lower = np.percentile(bootstrap_stats, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_stats, 97.5, axis=0)
        metric_names = ["Deviance"]
        results = {}
        for i, name in enumerate(metric_names):
            results[name] = {
                "Estimate": point_estimates[i],
                "95% CI": (ci_lower[i], ci_upper[i])
            }

        print(f"\nPrediction Errors for {self.best_model}:")
        for name, res in results.items():
            print(f"{name}: {res['Estimate']:.4f} (95% CI: {res['95% CI'][0]:.4f} - {res['95% CI'][1]:.4f})")
        return results

    def plot_feature_importance(self):
        if self.best_model is None or self.models is None:
            print("No best model has been identified. Train the model first.")
            return

        if not self.feature_columns:
            print("Feature names not provided.")
            return

        def compute_permutation_importance_torch(model, X, y, ids, feature_idx, device, n_repeats=10):
            model.eval()
            X_padded = self._pad_sequences(X)
            X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(device)
            with torch.no_grad():
                baseline_pred = model(X_tensor).cpu().numpy().flatten()
            baseline_dev = self._poisson_deviance(y, baseline_pred)
            scores = []

            for _ in range(n_repeats):
                X_permuted = [x.copy() for x in X]
                for i in range(len(X_permuted)):
                    np.random.shuffle(X_permuted[i][:, feature_idx])
                X_permuted_padded = self._pad_sequences(X_permuted)
                X_permuted_tensor = torch.tensor(X_permuted_padded, dtype=torch.float32).to(device)
                with torch.no_grad():
                    permuted_pred = model(X_permuted_tensor).cpu().numpy().flatten()
                permuted_dev = self._poisson_deviance(y, permuted_pred)
                scores.append(permuted_dev - baseline_dev)

            return np.mean(scores)

        model = self.models[self.best_model]
        feature_names = self.feature_columns
        n_features = len(feature_names)

        if self.best_model == 'REEM' and hasattr(model.fixed_model_, 'coef_'):
            importances = np.abs(model.fixed_model_.coef_)
            title = f"{self.best_model} (Absolute Coefficients)"
        elif self.best_model == 'RandomForest' and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            title = f"{self.best_model} (Feature Importances)"
        elif self.best_model == 'GPBoost' and hasattr(model.model, 'booster_'):
            booster = model.model.booster_
            importances = booster.feature_importance(importance_type='gain')
            if len(importances) != n_features:
                raise ValueError(f"GPBoost importance length ({len(importances)}) does not match feature count ({n_features})")
            title = f"{self.best_model} (Gain Importance)"
        else:
            importances = np.zeros(n_features)
            if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
                for i in range(n_features):
                    importances[i] = compute_permutation_importance_torch(
                        model, self.X_train, self.y_train, self.ids_train, i, self.device
                    )
            else:
                X_train_flat = np.vstack(self.X_train)
                y_train_expanded = np.concatenate([np.full(x.shape[0], y_val) for x, y_val in zip(self.X_train, self.y_train)])
                r = permutation_importance(
                    model, X_train_flat, y_train_expanded,
                    scoring=self._poisson_deviance_scorer, n_repeats=10, random_state=42
                )
                importances = r.importances_mean
            title = f"{self.best_model} (Permutation Importance)"

        importances = importances / np.max(np.abs(importances) + 1e-10)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title(title)
        plt.xlabel("Normalized Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def _poisson_deviance_scorer(self, estimator, X, y):
        if isinstance(X, list):
            X_flat = np.vstack(X)
        else:
            X_flat = X
        y_pred = estimator.predict(X)
        return -self._poisson_deviance(y, y_pred)

    def plot_predictions_vs_actual(self, X, y, ids=None):
        y_pred = self.predict(X, ids)
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title(f"Predictions vs Actual Counts ({self.best_model})")
        plt.xlabel("Actual Counts")
        plt.ylabel("Predicted Counts")
        plt.grid(True)
        plt.show()

    def train_and_evaluate(self, df_or_X=None, feature_columns=None, target_column=None, id_column=None, time_column=None, X=None, y=None, ids=None, batch_size=32, epochs=10, filepath="best_count_model.pkl"):
        if isinstance(df_or_X, pd.DataFrame):
            if any(arg is None for arg in [feature_columns, target_column, id_column, time_column]):
                raise ValueError("feature_columns, target_column, id_column, and time_column must be provided when passing a DataFrame.")
            X, y, ids = self.preprocess_data(df_or_X, feature_columns, target_column, id_column, time_column)
        else:
            if X is None or y is None or ids is None:
                raise ValueError("X, y, and ids must be provided when not passing a DataFrame.")
            X, y, ids = X, y, ids

        print("=== Longitudinal Count Regression ===")
        print("\nFitting model...")
        self.fit(X, y, ids, batch_size=batch_size, epochs=epochs)

        print("\nDisplaying outcome distribution...")
        self.display_outcome_distribution(y)

        print("\nGenerating predictions on full dataset...")
        preds = self.predict(X, ids)

        print("\nComputing prediction errors...")
        self.compute_prediction_errors(X, y, ids)

        print("\nPlotting feature importance...")
        self.plot_feature_importance()

        print("\nPlotting predictions vs actual...")
        self.plot_predictions_vs_actual(X, y, ids)

        print("\nSaving best model...")
        model = self.models[self.best_model]
        if self.best_model in ['TCN', 'LSTM', 'GRU', 'Transformer']:
            torch.save(model.state_dict(), filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        print(f"Best model ({self.best_model}) saved to {filepath}")

        print("\nTraining and evaluation completed.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import torch
from torch import nn
import torch.nn as nn
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.evaluation import EvalSurv
import torchtuples as tt
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLAgentSurvival:
    def __init__(self, data, time_col, event_col, feature_cols, test_size=0.2, random_state=42, models_to_fit=None):
        self.data = data
        self.time_col = time_col
        self.event_col = event_col
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.random_state = random_state
        self.models_to_fit = models_to_fit
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.labtrans = None  # For DeepHit and RNN
        self._prepare_data()

    def _prepare_data(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        X = self.data[self.feature_cols]
        y = self.data[self.event_col]
        
        for train_idx, test_idx in sss.split(X, y):
            self.X_train = X.iloc[train_idx].copy()
            self.X_test = X.iloc[test_idx].copy()
            self.y_train = self.data[[self.time_col, self.event_col]].iloc[train_idx].copy()
            self.y_test = self.data[[self.time_col, self.event_col]].iloc[test_idx].copy()
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train).astype(np.float32)
        self.X_test_scaled = self.scaler.transform(self.X_test).astype(np.float32)
        
        self.y_train_struct = np.array([(e, t) for e, t in zip(
            self.y_train[self.event_col], self.y_train[self.time_col])],
            dtype=[('event', bool), ('time', float)])
        self.y_test_struct = np.array([(e, t) for e, t in zip(
            self.y_test[self.event_col], self.y_test[self.time_col])],
            dtype=[('event', bool), ('time', float)])

    def plot_kaplan_meier(self, save_path='kaplan_meier.png', show_plot=False):
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(self.data[self.time_col], self.data[self.event_col])
            
            plt.figure(figsize=(10, 6))
            kmf.plot_survival_function()
            plt.title('Kaplan-Meier Survival Curve')
            plt.xlabel('Time')
            plt.ylabel('Survival Probability')
            plt.grid(True)
            plt.savefig(save_path)
            print(f"Kaplan-Meier plot saved to {save_path}")
            if show_plot:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"Error plotting Kaplan-Meier: {str(e)}")

    def fit_cox_ph(self):
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        cph = CoxPHFitter()
        cph.fit(train_data, duration_col=self.time_col, event_col=self.event_col)
        self.models['CoxPH'] = cph
        return cph

    def fit_rsf(self):
        rsf = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            random_state=self.random_state
        )
        rsf.fit(self.X_train_scaled, self.y_train_struct)
        self.models['RSF'] = rsf
        return rsf

    def fit_svm(self):
        svm = FastSurvivalSVM(random_state=self.random_state)
        svm.fit(self.X_train_scaled, self.y_train_struct)
        self.models['SVM'] = svm
        return svm

    def fit_deepsurv(self):
        model = Sequential([
            Dense(64, activation='relu', input_dim=len(self.feature_cols)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(self.X_train_scaled, self.y_train[self.time_col], 
                 epochs=50, batch_size=32, verbose=0)
        self.models['DeepSurv'] = model
        return model

    def fit_survival_cnn(self):
        X_train_reshaped = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], self.X_train_scaled.shape[1], 1)
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(len(self.feature_cols), 1)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(X_train_reshaped, self.y_train[self.time_col], 
                 epochs=50, batch_size=32, verbose=0)
        self.models['SurvivalCNN'] = model
        return model

    def fit_transformer(self):
        class SurvivalTransformer(nn.Module):
            def __init__(self, input_dim):
                super(SurvivalTransformer, self).__init__()
                nhead = min(4, max(1, input_dim // 2))
                while input_dim % nhead != 0:
                    nhead -= 1
                if nhead == 0:
                    nhead = 1
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead), num_layers=2)
                self.fc = nn.Linear(input_dim, 1)

            def forward(self, x):
                x = self.transformer(x)
                return self.fc(x[:, -1, :])

        model = SurvivalTransformer(len(self.feature_cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_train_tensor = torch.FloatTensor(self.X_train_scaled).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(self.y_train[self.time_col].values)
        
        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        self.models['Transformer'] = model
        return model

    def fit_deephit(self):
        self.labtrans = LabTransDiscreteTime(10)
        y_train = self.labtrans.fit_transform(self.y_train[self.time_col].values, 
                                             self.y_train[self.event_col].values)
        
        X_train_scaled_float32 = self.X_train_scaled.astype(np.float32)
        
        model = DeepHitSingle(
            net=tt.practical.MLPVanilla(
                in_features=len(self.feature_cols),
                num_nodes=[32, 32],
                out_features=self.labtrans.out_features,
                batch_norm=True,
                dropout=0.1
            ),
            duration_index=self.labtrans.cuts,
            alpha=0.2,
            sigma=0.1
        )
        model.optimizer.set_lr(0.01)
        model.fit(X_train_scaled_float32, y_train, batch_size=128, epochs=100, verbose=False)
        self.models['DeepHit'] = (model, self.labtrans)
        return model

    def fit_rnn(self):
        self.labtrans = LabTransDiscreteTime(10)
        y_train = self.labtrans.fit_transform(self.y_train[self.time_col].values, 
                                             self.y_train[self.event_col].values)
        
        X_train_scaled_float32 = self.X_train_scaled.astype(np.float32)
        X_train_reshaped = X_train_scaled_float32[:, np.newaxis, :]  # Shape: (n_samples, 1, n_features)
        
        class RNNNet(nn.Module):
            def __init__(self, input_size, num_nodes, output_size):
                super().__init__()
                self.rnn = nn.LSTM(input_size, hidden_size=num_nodes[0], batch_first=True)
                self.fc = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(num_nodes[0], output_size)
                )

            def forward(self, x):
                x, _ = self.rnn(x)
                return self.fc(x[:, -1, :])  # Last hidden state
        
        net = RNNNet(input_size=len(self.feature_cols), num_nodes=[64], output_size=self.labtrans.out_features)
        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=self.labtrans.cuts)
        model.optimizer.set_lr(0.01)
        model.fit(X_train_reshaped, y_train, batch_size=256, epochs=100, verbose=False)
        self.models['RNN'] = (model, self.labtrans)
        return model

    def cindex_scorer(self, estimator, X, y):
        """
        Compute the concordance index for scoring.
        """
        if hasattr(estimator, 'predict_surv_df'):  # DeepHit, RNN
            surv_df = estimator.predict_surv_df(X)
            ev = EvalSurv(surv_df, y[self.time_col].values, y[self.event_col].values, censor_surv='km')
            return ev.concordance_td('antolini')
        else:
            risk_score = estimator.predict(X)
            return concordance_index(y[self.time_col], -risk_score, y[self.event_col])

    def compute_metrics(self, model, name, X, y_time, y_event):
        try:
            y_time = np.array(y_time)
            y_event = np.array(y_event)
            
            times = np.percentile(y_time[y_time > 0], np.linspace(5, 95, 20))
            
            if name == 'CoxPH':
                cum_hazard = model.predict_cumulative_hazard(X, times=times)
                surv_prob = np.exp(-cum_hazard)  # Shape: (n_times, n_samples)
                y_pred = model.predict_partial_hazard(X)
                c_index = concordance_index(y_time, -y_pred, y_event)
            
            elif name == 'RSF':
                surv_func = model.predict_survival_function(X)
                surv_prob = np.array([sp(times) for sp in surv_func]).T  # Shape: (n_times, n_samples)
                y_pred = model.predict(X)
                c_index = concordance_index(y_time, -y_pred, y_event)
            
            elif name == 'SVM':
                y_pred = model.predict(X)
                surv_prob = np.ones((len(times), len(y_pred))) * np.exp(-y_pred)[None, :]  # Shape: (n_times, n_samples)
                c_index = concordance_index(y_time, -y_pred, y_event)
            
            elif name in ['DeepSurv', 'SurvivalCNN']:
                X_input = X if name == 'DeepSurv' else X.reshape(X.shape[0], X.shape[1], 1)
                y_pred = model.predict(X_input, verbose=0).flatten()
                surv_prob = np.ones((len(times), len(y_pred))) * np.exp(-y_pred)[None, :]  # Shape: (n_times, n_samples)
                c_index = concordance_index(y_time, -y_pred, y_event)
            
            elif name == 'Transformer':
                X_tensor = torch.FloatTensor(X).unsqueeze(1)
                y_pred = model(X_tensor).detach().numpy().flatten()
                surv_prob = np.ones((len(times), len(y_pred))) * np.exp(-y_pred)[None, :]  # Shape: (n_times, n_samples)
                c_index = concordance_index(y_time, -y_pred, y_event)
            
            elif name == 'DeepHit':
                model, labtrans = model
                model.interpolate(labtrans.out_features)
                surv_df = model.predict_surv_df(X)
                times = surv_df.index.values
                surv_prob = surv_df.values  # Shape: (n_times, n_samples)
                ev = EvalSurv(surv_df, y_time, y_event, censor_surv='km')
                c_index = ev.concordance_td('antolini')
                ibs = ev.integrated_brier_score(np.linspace(surv_df.index.min(), surv_df.index.max(), 100))
            
            elif name == 'RNN':
                model, labtrans = model
                X_reshaped = X[:, np.newaxis, :]  # Shape: (n_samples, 1, n_features)
                model.interpolate(labtrans.out_features)
                surv_df = model.predict_surv_df(X_reshaped)
                times = surv_df.index.values
                surv_prob = surv_df.values  # Shape: (n_times, n_samples)
                ev = EvalSurv(surv_df, y_time, y_event, censor_surv='km')
                c_index = ev.concordance_td('antolini')
                ibs = ev.integrated_brier_score(np.linspace(surv_df.index.min(), surv_df.index.max(), 100))
            
            if name in ['DeepHit', 'RNN']:
                # Time-dependent AUC for DeepHit and RNN
                min_test_time, max_test_time = y_time.min(), y_time.max()
                valid_times = times[(times >= min_test_time) & (times < max_test_time)]
                if len(valid_times) == 0:
                    print(f"No valid time points for AUC in {name}. Setting AUC to 0.")
                    auc_mean = 0.0
                else:
                    auc_scores = []
                    surv_probs_test = surv_prob.T  # Shape: (n_samples, n_times)
                    for t in valid_times:
                        col_idx = np.where(times == t)[0][0]
                        risk_scores = 1 - surv_probs_test[:, col_idx]
                        auc, _ = cumulative_dynamic_auc(
                            self.y_train_struct, self.y_test_struct, risk_scores, t
                        )
                        auc_scores.append(auc[0])
                    auc_mean = np.mean(auc_scores) if auc_scores else 0.0
                return {
                    'C-Index': round(c_index, 3),
                    'IBS': round(ibs, 3) if not np.isnan(ibs) else 0.0,
                    'Time-AUC': round(auc_mean, 3) if not np.isnan(auc_mean) else 0.0
                }
            
            # For non-DeepHit/RNN models
            eval_surv = EvalSurv(pd.DataFrame(surv_prob, index=times), y_time, y_event, censor_surv='km')
            ibs = eval_surv.integrated_brier_score(times)
            
            auc_scores = []
            for t in times:
                try:
                    risk_scores = -surv_prob[np.argmin(np.abs(times - t)), :]
                    auc, _ = cumulative_dynamic_auc(
                        self.y_train_struct, self.y_test_struct, risk_scores, t
                    )
                    auc_scores.append(auc[0])
                except:
                    continue
            auc_mean = np.mean(auc_scores) if auc_scores else 0.0
            
            return {
                'C-Index': round(c_index, 3),
                'IBS': round(ibs, 3) if not np.isnan(ibs) else 0.0,
                'Time-AUC': round(auc_mean, 3) if not np.isnan(auc_mean) else 0.0
            }
        except Exception as e:
            print(f"Error computing metrics for {name}: {str(e)}")
            return {'C-Index': 0.0, 'IBS': 0.0, 'Time-AUC': 0.0}

    def evaluate_all_models(self):
        for name, model in self.models.items():
            metrics = self.compute_metrics(
                model, name, 
                self.X_test_scaled, 
                self.y_test[self.time_col].values, 
                self.y_test[self.event_col].values
            )
            self.metrics[name] = metrics
            print(f"{name}: {metrics}")
        
        self.best_model = max(self.metrics.items(), key=lambda x: x[1]['C-Index'])[0]
        print(f"\nBest Model: {self.best_model}")

    def save_model(self, name, path):
        if name not in self.models:
            print(f"Model {name} not found.")
            return
        
        try:
            if name in ['Transformer', 'RNN']:
                model = self.models[name][0] if name == 'RNN' else self.models[name]
                torch.save(model.state_dict(), path)
                print(f"{name} saved to {path}")
            else:
                joblib.dump(self.models[name], path)
                print(f"{name} saved to {path}")
        except Exception as e:
            print(f"Error saving {name}: {str(e)}")

    def load_model(self, name, path):
        try:
            if name in ['Transformer', 'RNN']:
                if name == 'Transformer':
                    model = self.fit_transformer()  # Re-initialize structure
                    model.load_state_dict(torch.load(path))
                    self.models['Transformer'] = model
                else:
                    self.fit_rnn()  # Re-initialize structure
                    model, labtrans = self.models['RNN']
                    model.net.load_state_dict(torch.load(path))
                    self.models['RNN'] = (model, labtrans)
                print(f"{name} loaded from {path}")
            else:
                self.models[name] = joblib.load(path)
                print(f"{name} loaded from {path}")
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")

    def plot_best_model_feature_importance(self, save_path='feature_importance_best_model.png', show_plot=False):
        if not self.best_model:
            print("No best model selected. Run evaluate_all_models first.")
            return
        
        try:
            name = self.best_model
            model = self.models[name]
            plt.figure(figsize=(10, 8))
            
            if name == 'CoxPH':
                importance = pd.Series(model.params_, index=self.feature_cols)
                importance.sort_values().plot(kind='barh')
                plt.title(f'Feature Importance - Best Model ({name})')
                plt.xlabel('Coefficient')
            
            elif name == 'RSF':
                perm_importance = permutation_importance(
                    model,
                    self.X_test_scaled,
                    self.y_test,
                    scoring=self.cindex_scorer,
                    n_repeats=30,
                    random_state=self.random_state
                )
                importances_mean = perm_importance.importances_mean
                importances_std = perm_importance.importances_std
                
                features = np.array(self.feature_cols)
                sorted_idx = np.argsort(importances_mean)
                sorted_features = features[sorted_idx]
                sorted_importances = importances_mean[sorted_idx]
                sorted_std = importances_std[sorted_idx]
                
                plt.barh(range(len(sorted_importances)), sorted_importances, xerr=sorted_std,
                         align="center", color="skyblue")
                plt.yticks(range(len(sorted_importances)), sorted_features)
                plt.xlabel("Permutation Importance (C-index drop)")
                plt.title("Permutation-based Feature Importance for Random Survival Forest")
            
            elif name in ['DeepHit', 'RNN']:
                model_for_perm = model[0] if name in ['DeepHit', 'RNN'] else model
                X_test_df = pd.DataFrame(self.X_test_scaled, columns=self.feature_cols)
                baseline_surv_df = model_for_perm.predict_surv_df(X_test_df.values)
                ev = EvalSurv(baseline_surv_df, self.y_test[self.time_col].values, self.y_test[self.event_col].values, censor_surv='km')
                baseline_score = ev.concordance_td('antolini')
                
                importances = []
                for col in X_test_df.columns:
                    X_test_permuted = X_test_df.copy()
                    X_test_permuted[col] = np.random.permutation(X_test_permuted[col].values)
                    permuted_surv_df = model_for_perm.predict_surv_df(X_test_permuted.values)
                    permuted_ev = EvalSurv(permuted_surv_df, self.y_test[self.time_col].values, self.y_test[self.event_col].values, censor_surv='km')
                    permuted_score = permuted_ev.concordance_td('antolini')
                    importances.append(baseline_score - permuted_score)
                
                importances = np.array(importances)
                sorted_idx = np.argsort(importances)
                plt.barh(range(len(importances)), importances[sorted_idx])
                plt.yticks(range(len(importances)), np.array(self.feature_cols)[sorted_idx])
                plt.xlabel("Importance (decrease in C-index)")
                plt.title(f"Feature Importance (Permutation) for {name}")
            
            else:
                def scoring_function(model, X, y):
                    y_time, y_event = y[:, 0], y[:, 1]
                    metrics = self.compute_metrics(model, name, X, y_time, y_event)
                    return metrics['C-Index']
                
                X_test = self.X_test_scaled
                y_test = np.column_stack((self.y_test[self.time_col], self.y_test[self.event_col]))
                
                model_for_perm = model[0] if name in ['DeepHit', 'RNN'] else model
                result = permutation_importance(
                    model_for_perm, X_test, y_test, scoring=scoring_function, 
                    n_repeats=10, random_state=self.random_state
                )
                importance = pd.Series(result.importances_mean, index=self.feature_cols)
                importance.sort_values().plot(kind='barh')
                plt.title(f'Feature Importance - Best Model ({name})')
                plt.xlabel('Permutation Importance')
            
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Feature importance plot for best model ({name}) saved to {save_path}")
            if show_plot:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"Error plotting feature importance for best model ({name}): {str(e)}")

    def predict(self, new_data):
        X_new = new_data[self.feature_cols]
        X_new_scaled = self.scaler.transform(X_new).astype(np.float32)
        
        model = self.models[self.best_model]
        
        if self.best_model == 'CoxPH':
            predictions = model.predict_partial_hazard(X_new)
        
        elif self.best_model in ['RSF', 'SVM']:
            predictions = model.predict(X_new_scaled)
        
        elif self.best_model == 'DeepSurv':
            predictions = model.predict(X_new_scaled, verbose=0).flatten()
        
        elif self.best_model == 'SurvivalCNN':
            X_new_reshaped = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
            predictions = model.predict(X_new_reshaped, verbose=0).flatten()
        
        elif self.best_model == 'Transformer':
            X_new_tensor = torch.FloatTensor(X_new_scaled).unsqueeze(1)
            predictions = model(X_new_tensor).detach().numpy().flatten()
        
        elif self.best_model == 'DeepHit':
            model, _ = model
            predictions = model.predict_surv_df(X_new_scaled).iloc[-1].values
        
        elif self.best_model == 'RNN':
            model, _ = model
            X_new_reshaped = X_new_scaled[:, np.newaxis, :]  # Shape: (n_samples, 1, n_features)
            predictions = model.predict_surv_df(X_new_reshaped).iloc[-1].values
        
        result = X_new.copy()
        result['Prediction'] = predictions
        return result

    def fit_all_models(self):
        valid_models = ['CoxPH', 'RSF', 'SVM', 'DeepSurv', 'SurvivalCNN', 'Transformer', 'DeepHit', 'RNN']
        models_to_fit = self.models_to_fit if self.models_to_fit is not None else valid_models
        
        if self.models_to_fit is not None:
            invalid_models = [m for m in models_to_fit if m not in valid_models]
            if invalid_models:
                raise ValueError(f"Invalid model names: {invalid_models}. Choose from {valid_models}")
        
        if 'CoxPH' in models_to_fit:
            self.fit_cox_ph()
        if 'RSF' in models_to_fit:
            self.fit_rsf()
        if 'SVM' in models_to_fit:
            self.fit_svm()
        if 'DeepSurv' in models_to_fit:
            self.fit_deepsurv()
        if 'SurvivalCNN' in models_to_fit:
            self.fit_survival_cnn()
        if 'Transformer' in models_to_fit:
            self.fit_transformer()
        if 'DeepHit' in models_to_fit:
            self.fit_deephit()
        if 'RNN' in models_to_fit:
            self.fit_rnn()
        
        if not self.models:
            raise ValueError("No models were fitted. Please specify valid models in models_to_fit.")
        
        self.evaluate_all_models()


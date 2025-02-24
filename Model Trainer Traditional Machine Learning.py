from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tqdm import tqdm

def split_data(df, target_column, test_size=0.2, val_size=0.25, random_state=42):
    """
    Splits the DataFrame into training, validation, and test sets.

    Parameters:
        df (DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training+validation set to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 1: Split into training+validation and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Split the training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def define_pipelines_and_hyperparams():
    """
    Define pipelines and hyperparameter grids for various classifiers.

    Returns:
        dict: A dictionary containing pipelines and their respective hyperparameter grids.
    """
    # Define pipelines
    pipe_mlp = Pipeline([
        ('clf', MLPClassifier(random_state=42))
    ])

    pipe_logistic = Pipeline([
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipe_rf = Pipeline([
        ('clf', RandomForestClassifier(random_state=42))
    ])

    pipe_gbm = Pipeline([
        ('clf', GradientBoostingClassifier(random_state=42))
    ])

    # Logistic Regression Hyperparameter Grid
    param_dist_logistic = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2', 'none'],
        'clf__solver': ['lbfgs', 'saga'],
        'clf__max_iter': [100, 200, 500]
    }
    param_dist_mlp = {
        'clf__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'clf__activation': ['tanh', 'relu'],
        'clf__solver': ['adam', 'sgd'],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate': ['constant', 'adaptive'],
        'clf__max_iter': [200, 300, 500]
    }

    param_dist_rf = {
        'clf__n_estimators': [100, 200, 500],
        'clf__max_depth': [5, 10, 20, 50, None],
        'clf__min_samples_split': [2, 5, 10, 50],
        'clf__min_samples_leaf': [1, 2, 5, 10],
        'clf__max_features': ['sqrt', 'log2', None]
    }

    param_dist_gbm = {
        'clf__n_estimators': [100, 200, 300, 500],
        'clf__learning_rate': [0.001, 0.01, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7, 10],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__subsample': [0.7, 0.8, 0.9, 1.0],
        'clf__max_features': ['sqrt', 'log2', None]
    }

    return {
        'logistic': (pipe_logistic, param_dist_logistic),
        'mlp': (pipe_mlp, param_dist_mlp),
        'random_forest': (pipe_rf, param_dist_rf),
        'gradient_boosting': (pipe_gbm, param_dist_gbm)
    }

def run_randomized_search(pipe, param_dist, X_train, y_train, X_val, y_val):
    """
    Perform randomized search for hyperparameter tuning.

    Parameters:
        pipe (Pipeline): The pipeline for the classifier.
        param_dist (dict): The hyperparameter grid.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_val (DataFrame): Validation features.
        y_val (Series): Validation labels.

    Returns:
        estimator: The best estimator from the search.
    """
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        scoring='roc_auc',
        n_iter=10,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = search.best_estimator_.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print(f"Best Params: {search.best_params_}")
    print(f"Best Cross-Validated AUC: {search.best_score_}")
    print(f"Validation AUC: {val_auc}")

    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and calculate all metrics and curves.

    Parameters:
        model: Trained model.
        X_test (DataFrame): Test features.
        y_test (Series): Test labels.

    Returns:
        None
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

def main(df, target_column):
    """
    Main function to run hyperparameter tuning and evaluation.
    """
    print(" Splitting Data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column)

    print(" Defining Models and Hyperparameters...")
    pipelines_and_params = define_pipelines_and_hyperparams()

    best_models = {}

    print("\n Hyperparameter Tuning for Each Model:")
    for model_name, (pipe, param_dist) in tqdm(
        pipelines_and_params.items(), desc="Tuning Models", unit="model"
    ):
        print(f"\n {model_name.capitalize()}:")
        best_models[model_name] = run_randomized_search(pipe, param_dist, X_train, y_train, X_val, y_val)

    print("\n Evaluating Best Models on Test Set:")
    for model_name, model in tqdm(
        best_models.items(), desc="Evaluating Models", unit="model"
    ):
        print(f"\n {model_name.capitalize()}:")
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    df  = pd.read_csv('df_1_13_2025.csv', index_col=False)
    main(df, target_column="retention")


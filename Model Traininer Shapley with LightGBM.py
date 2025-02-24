from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMClassifier
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector, ShapPlotFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def train_and_select_features_with_shap(X_train, y_train, X_test, y_test, num_features=20, n_iter_shap=5, n_iter_random_search=50):
    """
    Train a LightGBM model, perform feature selection using Zoish's ShapFeatureSelector, 
    tune hyperparameters using RandomizedSearchCV, and evaluate the best model.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): Test labels.
        num_features (int): Number of features to retain during feature selection.
        n_iter_shap (int): Number of iterations for SHAP feature selection.
        n_iter_random_search (int): Number of iterations for RandomizedSearchCV.

    Returns:
        dict: A dictionary containing evaluation metrics and the top features list.
    """
    # Step 1: Train the LightGBM model
    lgb_model = LGBMClassifier(n_estimators=100, random_state=42, max_depth=6)
    lgb_model.fit(X_train, y_train)

    # Step 2: Perform SHAP-based feature selection
    selector = ShapFeatureSelector(
        model=lgb_model,
        num_features=num_features,
        n_iter=n_iter_shap,
        direction="maximum",
        scoring="roc_auc",
        cv=5
    )

    selector.fit(X_train, y_train)

    # Transform the data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Selected number of features: {X_train_selected.shape[1]}")
    

    # Step 3: Hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 40, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    model = LGBMClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        scoring='roc_auc',
        n_iter=n_iter_random_search,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_selected, y_train)

    # Step 4: Evaluate the best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_selected)
    y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nClassification Metrics on Test Data:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    # Step 5: Visualize SHAP feature importance
    plot_factory = ShapPlotFeatures(selector)
    plot_factory.summary_plot()
    plot_factory.bar_plot()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
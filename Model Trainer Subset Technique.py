from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

# Function to check predict_proba support
def safe_predict_proba(clf, X_test):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        return clf.decision_function(X_test)
    else:
        raise ValueError(f"{clf.__class__.__name__} does not support probability predictions.")

# Function to check multicollinearity (for Logistic Regression)
def check_multicollinearity(X):
    corr_matrix = pd.DataFrame(X).corr().abs()
    high_corr = (corr_matrix > 0.9).sum().sum() - len(corr_matrix)
    if high_corr > 0:
        warnings.warn("High multicollinearity detected. Logistic Regression results might be unstable.")

# Function to train and evaluate a single subset
def process_subset(args):
    i, split, classifiers, negative, target_column, random_state = args
    subset = pd.concat([split, negative])
    X = subset.drop(columns=[target_column])
    y = subset[target_column]

    # Check multicollinearity for Logistic Regression
    if 'Logistic Regression' in classifiers:
        check_multicollinearity(X)

    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    results = {}
    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = safe_predict_proba(clf, X_test)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                "Accuracy": report["accuracy"],
                "Precision (1)": report["1"]["precision"],
                "Recall (1)": report["1"]["recall"],
                "F1-score (1)": report["1"]["f1-score"],
                "AUC-ROC": auc_roc
            }
        except Exception as e:
            results[name] = {"Error": str(e)}
    return {f'Subset {i + 1}': results}

# Define reusable function for subsetting and evaluation
def evaluate_subsets(df, target_column, classifiers, splits=3, random_state=42):
    """
    Evaluate models across multiple subsets of positive class data.

    Parameters:
        df (DataFrame): The dataset with features and target.
        target_column (str): The name of the target column.
        classifiers (dict): A dictionary of classifiers to evaluate.
        splits (int): Number of splits for the positive class.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrames containing subset metrics and averaged metrics.
    """
    # Separate positive and negative classes
    positive = df[df[target_column] == 1]
    negative = df[df[target_column] == 0]

    # Split positive class into equal portions
    split_size = len(positive) // splits
    remainder = len(positive) % splits
    positive_splits = [
        positive.iloc[i * split_size + min(i, remainder):(i + 1) * split_size + min(i + 1, remainder)]
        for i in range(splits)
    ]

    # Prepare arguments for multiprocessing
    args = [
        (i, split, classifiers, negative, target_column, random_state)
        for i, split in enumerate(positive_splits)
    ]

    # Run evaluation in parallel using multiprocessing
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_subset, args), total=len(args)))

    # Combine results
    all_results = {k: v for d in results for k, v in d.items()}

    # Calculate average metrics
    averaged_metrics = {}
    metric_keys = ["Accuracy", "Precision (1)", "Recall (1)", "F1-score (1)", "AUC-ROC"]

    for model in classifiers.keys():
        model_metrics = {key: [] for key in metric_keys}

        for subset, metrics in all_results.items():
            if model in metrics and "Error" not in metrics[model]:
                for key in metric_keys:
                    model_metrics[key].append(metrics[model][key])

        averaged_metrics[model] = {key: np.mean(values) for key, values in model_metrics.items() if values}

    # Convert individual subset metrics to a DataFrame
    subset_metrics_list = []

    for subset, metrics in all_results.items():
        for model, report in metrics.items():
            if "Error" not in report:
                report['Model'] = model
                report['Subset'] = subset
                subset_metrics_list.append(report)

    subset_metrics_df = pd.DataFrame(subset_metrics_list)

    # Convert averaged metrics to a DataFrame
    averaged_metrics_list = []

    for model, metrics in averaged_metrics.items():
        metrics['Model'] = model
        averaged_metrics_list.append(metrics)

    averaged_metrics_df = pd.DataFrame(averaged_metrics_list)

    return subset_metrics_df, averaged_metrics_df

# Example Usage:
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('../Data/df_1_13_2025.csv', index_col =False)

     
    subset_metrics, averaged_metrics = evaluate_subsets(
    df=df,
    target_column='retention',
    classifiers={
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=500, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            max_features='sqrt', 
            max_depth=20, 
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42, 
            subsample=0.9, 
            n_estimators=200, 
            min_samples_split=5, 
            min_samples_leaf=5, 
            max_features='sqrt', 
            max_depth=5, 
            learning_rate=0.2
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            solver='saga', 
            penalty='l1', 
            max_iter=500, 
            C=1
        ),
        'MLP': MLPClassifier(
            random_state=42, 
            solver='adam', 
            max_iter=500, 
            learning_rate='constant', 
            hidden_layer_sizes=(100, 50), 
            alpha=0.0001, 
            activation='relu'
        )
        },
        splits=3,
        random_state=42
    )
    
    print(subset_metrics)
    print(averaged_metrics)
    
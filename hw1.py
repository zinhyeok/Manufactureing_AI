import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target (0: non-defective, 1: defective)
    return X, y

def stratified_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def compute_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    AUC = roc_auc_score(y_true, y_pred)
    
    return TPR, FPR, AUC

def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'{title} (AUC = {auc_score:.2f})')

def adaboost(X_train, y_train, X_test, y_test, T=50):
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alpha_values = []
    
    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train, sample_weight=weights)
        y_pred_train = clf.predict(X_train)
        err = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)
        alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
        
        weights *= np.exp(-alpha * y_train * y_pred_train)
        weights /= np.sum(weights)
        
        classifiers.append(clf)
        alpha_values.append(alpha)
    
    def predict(X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(alpha_values, classifiers))
        return np.sign(final_pred)
    
    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)
    
    plot_roc_curve(y_test, y_test_pred, "AdaBoost")
    return compute_metrics(y_train, y_train_pred), compute_metrics(y_test, y_test_pred)

def adacost(X_train, y_train, X_test, y_test, T=50, cost=2.0):
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alpha_values = []
    
    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train, sample_weight=weights)
        y_pred_train = clf.predict(X_train)
        err = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)
        alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
        
        weight_update = np.exp(-alpha * y_train * y_pred_train)
        weight_update[y_train == 1] **= cost  # Increase penalty for misclassified defective items
        weights *= weight_update
        weights /= np.sum(weights)
        
        classifiers.append(clf)
        alpha_values.append(alpha)
    
    def predict(X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(alpha_values, classifiers))
        return np.sign(final_pred)
    
    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)
    
    plot_roc_curve(y_test, y_test_pred, "AdaCost")
    return compute_metrics(y_train, y_train_pred), compute_metrics(y_test, y_test_pred)

def adaauc(X_train, y_train, X_test, y_test, T=50):
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alpha_values = []
    
    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train, sample_weight=weights)
        y_pred_train = clf.predict(X_train)
        err = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)
        
        alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
        auc_score = roc_auc_score(y_train, y_pred_train)
        alpha *= (1 - auc_score)  # Modify alpha based on AUC
        
        weights *= np.exp(-alpha * y_train * y_pred_train)
        weights /= np.sum(weights)
        
        classifiers.append(clf)
        alpha_values.append(alpha)
    
    def predict(X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(alpha_values, classifiers))
        return np.sign(final_pred)
    
    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)
    
    plot_roc_curve(y_test, y_test_pred, "AdaAUC")
    return compute_metrics(y_train, y_train_pred), compute_metrics(y_test, y_test_pred)

# Load dataset
X, y = load_data('battery.csv')
X_train, X_test, y_train, y_test = stratified_split(X, y)

# Train models
plt.figure(figsize=(8, 6))
adaboost_results = adaboost(X_train, y_train, X_test, y_test)
adacost_results = adacost(X_train, y_train, X_test, y_test, cost=10.0)
adaauc_results = adaauc(X_train, y_train, X_test, y_test)
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Boosting Algorithms")
plt.show()

# Display results
print("AdaBoost Results (Train, Test):", adaboost_results)
print("AdaCost Results (Train, Test):", adacost_results)
print("AdaAUC Results (Train, Test):", adaauc_results)

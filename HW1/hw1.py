import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# 데이터 로드 및 전처리
def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target (0: non-defective, 1: defective)
    return X, y

def stratified_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def compute_metrics(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc

def plot_roc_curve(ax, y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    ax.plot(fpr, tpr, label=f'{title} (AUC = {auc_score:.2f})')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()

def plot_depth_changes(ax, depths, title):
    ax.plot(range(1, len(depths) + 1), depths, marker='o', label=title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tree Depth")
    ax.set_title(title)
    ax.legend()

class AdaBoost:
    def __init__(self, T=50, max_depth=5):
        self.T = T
        self.max_depth = max_depth
        self.classifiers = []
        self.alpha_values = []
        self.depths = []
    
    def train(self, X_train, y_train):
        n_samples = X_train.shape[0]
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.T):
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X_train, y_train, sample_weight=weights)
            y_pred_train = clf.predict(X_train)
            err = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            
            weights *= np.exp(-alpha * y_train * y_pred_train)
            weights /= np.sum(weights)
            
            self.classifiers.append(clf)
            self.alpha_values.append(alpha)
            self.depths.append(clf.get_depth())
    
    def predict(self, X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alpha_values, self.classifiers))
        return np.sign(final_pred)

    def evaluate(self, X_test, y_test):
        y_test_pred = self.predict(X_test)
        return compute_metrics(y_test, y_test_pred)

class AdaCost:
    def __init__(self, T=50, max_depth=5, cost=1.0):
        self.T = T
        self.max_depth = max_depth
        self.cost = cost
        self.classifiers = []
        self.alpha_values = []
        self.depths = []

    def train(self, X_train, y_train):
        n_samples = X_train.shape[0]
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.T):
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X_train, y_train, sample_weight=weights)
            y_pred_train = clf.predict(X_train)
            err = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            
            weight_update = np.exp(-alpha * y_train * y_pred_train)
            weight_update[y_train == 1] **= self.cost
            weights *= weight_update
            weights /= np.sum(weights)
            
            self.classifiers.append(clf)
            self.alpha_values.append(alpha)
            self.depths.append(clf.get_depth())
    
    def predict(self, X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alpha_values, self.classifiers))
        return np.sign(final_pred)

    def evaluate(self, X_test, y_test):
        y_test_pred = self.predict(X_test)
        return compute_metrics(y_test, y_test_pred)

class AdaAUC:
    def __init__(self, T=50, max_depth=5):
        self.T = T
        self.max_depth = max_depth
        self.classifiers = []
        self.alpha_values = []
        self.depths = []

    def train(self, X_train, y_train):
        n_samples = X_train.shape[0]
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.T):
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X_train, y_train, sample_weight=weights)
            y_pred_train = clf.predict(X_train)
            err = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            auc_score = roc_auc_score(y_train, y_pred_train)
            alpha *= (1 - auc_score)
            
            weights *= np.exp(-alpha * y_train * y_pred_train)
            weights /= np.sum(weights)
            
            self.classifiers.append(clf)
            self.alpha_values.append(alpha)
            self.depths.append(clf.get_depth())
    
    def predict(self, X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alpha_values, self.classifiers))
        return np.sign(final_pred)

    def evaluate(self, X_test, y_test):
        y_test_pred = self.predict(X_test)
        return compute_metrics(y_test, y_test_pred)

# 데이터 불러오기
X, y = load_data('battery.csv')
X_train, X_test, y_train, y_test = stratified_split(X, y)

# 모델 생성 및 학습
adaboost = AdaBoost()
adaboost.train(X_train, y_train)

adacost = AdaCost(cost=2.0)
adacost.train(X_train, y_train)

adaauc = AdaAUC()
adaauc.train(X_train, y_train)

# 평가
adaboost_auc = adaboost.evaluate(X_test, y_test)
adacost_auc = adacost.evaluate(X_test, y_test)
adaauc_auc = adaauc.evaluate(X_test, y_test)

# 그래프 출력
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
ax1 = axes[0]
ax2 = axes[1]

plot_roc_curve(ax1, y_test, adaboost.predict(X_test), "AdaBoost")
plot_roc_curve(ax1, y_test, adacost.predict(X_test), "AdaCost")
plot_roc_curve(ax1, y_test, adaauc.predict(X_test), "AdaAUC")

plot_depth_changes(ax2, adaboost.depths, "AdaBoost Depth")
plot_depth_changes(ax2, adacost.depths, "AdaCost Depth")
plot_depth_changes(ax2, adaauc.depths, "AdaAUC Depth")

plt.show()

# 결과 출력
print("AdaBoost AUC:", adaboost_auc)
print("AdaCost AUC:", adacost_auc)
print("AdaAUC AUC:", adaauc_auc)
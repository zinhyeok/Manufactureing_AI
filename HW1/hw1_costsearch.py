import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin

# 데이터 로드 및 전처리
def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target (0: non-defective, 1: defective)
    return X, y

def stratified_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def compute_metrics(y_true, y_pred):
    # TP = np.sum((y_true == 1) & (y_pred == 1))
    # FN = np.sum((y_true == 1) & (y_pred == 0))
    # TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    # return TPR
    auc = roc_auc_score(y_true, y_pred)
    return auc

class AdaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, T=50, max_depth=5, cost=1.0):
        self.T = T
        self.max_depth = max_depth
        self.cost = cost
        self.classifiers = []
        self.alpha_values = []
        self.depths = []

    def fit(self, X_train, y_train):
        self.classifiers = []
        self.alpha_values = []
        self.depths = []
        
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
        return self
    
    def predict(self, X):
        final_pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alpha_values, self.classifiers))
        return np.sign(final_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return compute_metrics(y, y_pred)

# 데이터 불러오기
X, y = load_data('battery.csv')
X_train, X_test, y_train, y_test = stratified_split(X, y)

# GridSearchCV를 사용하여 최적의 cost 값 찾기
param_grid = {'cost': np.linspace(1, 5, 5)}

grid_search = GridSearchCV(estimator=AdaCost(), param_grid=param_grid, scoring=make_scorer(compute_metrics), cv=3)
grid_search.fit(X_train, y_train)

best_cost = grid_search.best_params_['cost']
print(f'Best Cost for AdaCost: {best_cost}')

# 최적 cost로 모델 학습 및 평가
adacost_best = AdaCost(cost=best_cost)
adacost_best.fit(X_train, y_train)
best_tpr = adacost_best.score(X_test, y_test)

print(f'Best TPR with Cost {best_cost}: {best_tpr}')

# TPR vs Cost Plot
plt.figure(figsize=(8, 6))
plt.plot(param_grid['cost'], grid_search.cv_results_['mean_test_score'], marker='o', linestyle='-', label='AUC vs Cost')
plt.xlabel("Cost Parameter")
plt.ylabel("Area Under Curve(AUC)")
plt.title("Effect of Cost Parameter on AUC")
plt.legend()
plt.show()
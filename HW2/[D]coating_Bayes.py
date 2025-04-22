import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from gp import GaussianProcess  
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
# Online adaptation scenario with Bayesian Optimization (BO)
# Regression with Gaussian Process (GP) for Bayesian Optimization (BO)
# Note: Data differs by coil

import numpy as np
import pandas as pd
from gp import GaussianProcess
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === 1. Load Data ===
df = pd.read_csv("coating_sampled.csv")
coil_change_idx = df[df["coil"] != df["coil"].shift(1)].index.tolist()
start_idx = coil_change_idx[1]  # CRG1588 시작점 (index 16)
df_new = df.iloc[start_idx:].reset_index(drop=True)

# === 2. 초기 관측치 선택 ===
init_n = 2
init_df = df_new.iloc[:init_n]
X_env = ['thickness', 'width', 'speed', 'tension']
X_ctrl = ['gap_top', 'pressure_top']
y_col = 'weight_top'

# === 3. 데이터셋 구성 ===
X_train = []
y_train = []


for i in range(1, init_n):  # t-1 환경 사용
    x_env = init_df.iloc[i - 1][X_env].values
    x_ctrl = init_df.iloc[i][X_ctrl].values
    x_input = np.concatenate([x_env, x_ctrl])
    X_train.append(x_input)
    y_train.append(init_df.iloc[i][y_col])
X_train = np.array(X_train)
y_train = np.array(y_train)

# === 4. GP 모델 학습 ===
gp = GaussianProcess(kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2))
gp.fit(X_train, y_train)
gp.optimize_hyperparameters(bounds=((1e-2, 10.0), (1e-2, 10.0)))

# === 5. Expected Improvement 구현 ===
def expected_improvement(x, model, x_env_fixed, y_best):
    x_input = np.concatenate([x_env_fixed, x])
    mu, cov = model.predict(x_input.reshape(1, -1))
    sigma = np.sqrt(np.diag(cov))[0]
    if sigma == 0:
        return 0
    z = (y_best - mu[0]) / sigma
    ei = (y_best - mu[0]) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei if mu[0] >= df_new['target'].values[0] else 0


def suggest_next_point(gp_model, x_env_fixed, y_best, bounds, n_restarts=5):
    best_x = None
    best_ei = -np.inf

    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        res = minimize(lambda x: -expected_improvement(x, gp_model, x_env_fixed, y_best),
                       x0=x0, bounds=bounds, method='L-BFGS-B')
        ei_val = -res.fun
        if ei_val > best_ei:
            best_ei = ei_val
            best_x = res.x
    return best_x

# === 6. 최적화 loop 수행 ===
from scipy.stats import norm
n_iter = 9
bo_log = []

for t in range(n_iter):
    # 현재 환경변수는 마지막 관측치의 것 사용 (t-1 방식)
    x_env = df_new.iloc[init_n + t - 1][X_env].values
    y_target = df_new.iloc[init_n + t][['target']].values[0]

    bounds = [(7.5, 10.0), (0.2, 0.35)]  # EDA 기반
    x_next_ctrl = suggest_next_point(gp, x_env, y_best=min(y_train), bounds=bounds)

    # 다음 제안으로 예측
    x_input = np.concatenate([x_env, x_next_ctrl])
    y_pred, _ = gp.predict(x_input.reshape(1, -1))

    # 관측값 사용
    y_obs = df_new.iloc[init_n + t][y_col]

    # 데이터 업데이트
    X_train = np.vstack([X_train, x_input])
    y_train = np.append(y_train, y_obs)
    gp.fit(X_train, y_train)

    bo_log.append({
        "iteration": t,
        "gap_top": x_next_ctrl[0],
        "pressure_top": x_next_ctrl[1],
        "predicted_weight": y_pred[0],
        "observed_weight": y_obs,
        "target": y_target,
        "abs_error": abs(y_obs - y_target),
        "init_n": init_n + t,
    })

# === 7. 로그 확인 ===
bo_df = pd.DataFrame(bo_log)
print(bo_df)

# 실제 CRG1588 코일의 관측된 gap_top과 pressure_top
df_crg1588 = df[df["coil"] == " CRG1588"]
true_points = df_crg1588[["gap_top", "pressure_top"]].values

# BO 제안 점들을 init_n 기준으로 분리
bo_groups = bo_df.groupby("init_n")

# 시각화
plt.figure(figsize=(10, 8))

# 실제 값
plt.scatter(true_points[:, 0], true_points[:, 1], alpha=0.3, label="Actual (CRG1588)", color="gray", s=60)

# BO 제안 점들
colors = plt.cm.viridis(np.linspace(0, 1, len(bo_groups)))
for i, (init_n, group_df) in enumerate(bo_groups):
    plt.scatter(group_df["gap_top"], group_df["pressure_top"], 
                color=colors[i], label=f"BO (init_n={init_n})", s=100, marker="x")

plt.xlabel("gap_top")
plt.ylabel("pressure_top")
plt.title("BO Control Suggestions vs Actual (CRG1588)")
plt.legend()
plt.grid(True)
plt.show()

# # 2. predicted vs observed coating weight
#predicted vs observed vs target (weight)
plt.figure(figsize=(12, 6))

# lineplot은 hue 한 번만 사용, 각각 style로 구분
sns.lineplot(data=bo_df, x="init_n", y="observed_weight", linestyle="dotted", label="Observed (Actual)")
sns.lineplot(data=bo_df, x="init_n", y="predicted_weight", linestyle="dashed", label="Predicted (BO)")
plt.axhline(y=df_new["target"].iloc[0], color="red", linestyle="--", label="Target (Constant)")

plt.title("Predicted vs Observed vs Target Coating Weight")
plt.ylabel("Weight (Top-side)")
plt.grid(True)
plt.legend()
plt.show()

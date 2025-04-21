import numpy as np
import pandas as pd
from scipy.stats import norm
from gp import GaussianProcess  
import matplotlib.pyplot as plt
#online 상황가정
# Regression with Gaussian Process (GP) for Bayesian Optimization (BO)
# Data가 코일별로 다르다는점에 유의할것 

# 1. 데이터 로딩 및 초기 설정
df = pd.read_csv("coating_sampled.csv")
coil_names = df["coil"].unique()
new_coil_name = coil_names[1]  # CRG2188 이후의 새 coil
new_coil_df = df[df["coil"] == new_coil_name].reset_index(drop=True)
#' CRG1588'


# 2. Search Space 설정
init_N = 5
init_data = new_coil_df.iloc[:init_N]


X_env_init = init_data[["thickness", "width", "speed", "tension", "angle_top"]].values
X_ctrl_init = init_data[["gap_top", "pressure_top"]].values
X_train = np.concatenate([X_env_init, X_ctrl_init], axis=1)
y_train = init_data["weight_top"].values
y_target = new_coil_df["target"].iloc[0] 

# 2. EI 정의
def expected_improvement(x_ctrl, x_env_fixed, gp, y_target):
    x_input = np.concatenate([
        x_env_fixed.reshape(1, -1), 
        x_ctrl.reshape(1, -1)
    ], axis=1).astype(np.float64)

    mu, cov = gp.predict(x_input)
    mu = mu[0]
    sigma = np.sqrt(cov[0, 0]) if cov.ndim > 1 else np.sqrt(cov)

    if sigma < 1e-6:
        return max(0.0, mu - y_target)

    z = (mu - y_target) / sigma
    ei = (mu - y_target) * norm.cdf(z) + sigma * norm.pdf(z)

    return ei if mu >= y_target else 0.0  # ✅ target 이하일 경우 EI = 0


# 3. Acquisition Function 최대화 (Grid Search)
def suggest_next(gp, x_env_fixed, y_target):
    gap_grid = np.linspace(0.5, 2.5, 30)
    pressure_grid = np.linspace(0.5, 2.5, 30)
    best_ei = -np.inf
    best_x = None

    for g in gap_grid:
        for p in pressure_grid:
            ei = expected_improvement(
                np.array([g, p]), x_env_fixed, gp, y_target
            )
            if ei > best_ei:
                best_ei = ei
                best_x = np.array([g, p])
    return best_x


# 4. BO Loop
BO_iter = 5
gp = GaussianProcess(kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2))

for t in range(init_N, init_N + BO_iter):
    x_env_t = new_coil_df.iloc[t - 1][["thickness", "width", "speed", "tension", "angle_top"]].values.astype(np.float64)
    y_target_t = new_coil_df.iloc[t]["target"]  # 시점별 target

    gp.fit(X_train, y_train)
    gp.optimize_hyperparameters(bounds=((1e-2, 10.0), (1e-2, 10.0)))

    x_next_ctrl = suggest_next(gp, x_env_t, y_target_t)
    x_next_full = np.concatenate([x_env_t, x_next_ctrl])

    y_next, _ = gp.predict(x_next_full.reshape(1, -1))
    y_next = y_next[0]

    X_train = np.vstack([X_train, x_next_full])
    y_train = np.append(y_train, y_next)


# 5. 결과 DataFrame
results_df = pd.DataFrame(X_train[:, -2:], columns=["gap", "pressure"])
results_df["predicted_weight"] = y_train
results_df["loss"] = np.abs(results_df["predicted_weight"] - y_target)

print(results_df)
plt.plot(results_df["predicted_weight"], marker="o", label="Predicted Weight")
plt.axhline(y=y_target, color="r", linestyle="--", label="Target Weight")
plt.xlabel("Iteration")
plt.ylabel("Weight")
plt.legend()
plt.title("BO Iteration Trace")
plt.grid(True)
plt.show()
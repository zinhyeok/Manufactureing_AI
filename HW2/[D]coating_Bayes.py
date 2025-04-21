import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from gp import GaussianProcess  
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Online adaptation scenario with Bayesian Optimization (BO)
# Regression with Gaussian Process (GP) for Bayesian Optimization (BO)
# Note: Data differs by coil

# 1. Load data and identify transition point
df = pd.read_csv("coating_sampled.csv")
transition_index = df[df["coil"].str.strip() == "CRG2188"].index[-1] + 1
second_coil_name = df.iloc[transition_index]["coil"].strip()
second_coil_df = df[df["coil"].str.strip() == second_coil_name].reset_index(drop=True)

# 2. Select initial observations from the second coil
initial_obs_count = 5
init_data_second_coil = second_coil_df.iloc[:initial_obs_count]

env_features = ["thickness", "width", "speed", "tension"]
ctrl_features = ["gap_top", "pressure_top"]

X_env_init_second = init_data_second_coil[env_features].values
X_ctrl_init_second = init_data_second_coil[ctrl_features].values
y_train_second = init_data_second_coil["weight_top"].values

# Scale environmental variables
scaler_env = StandardScaler()
X_env_scaled_second = scaler_env.fit_transform(X_env_init_second)
X_train_second = np.concatenate([X_env_scaled_second, X_ctrl_init_second], axis=1)

# 3. Define Expected Improvement (EI)
def expected_improvement(x_ctrl, x_env_scaled, gp, y_target):
    x_input = np.concatenate([x_env_scaled.reshape(1, -1), x_ctrl.reshape(1, -1)], axis=1).astype(np.float64)
    mu, cov = gp.predict(x_input)
    mu = mu[0]
    sigma = np.sqrt(cov[0, 0]) if cov.ndim > 1 else np.sqrt(cov)
    if sigma < 1e-6:
        return 0.0
    z = (mu - y_target) / sigma
    ei = (mu - y_target) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

# 4. Suggest next control settings using L-BFGS-B
def suggest_next_LBFGSB(gp, x_env_unscaled, y_target):
    x_env_scaled = scaler_env.transform(x_env_unscaled.reshape(1, -1))[0]

    # Objective: -EI (maximize EI == minimize -EI)
    def neg_ei(x_ctrl):
        return -expected_improvement(np.array(x_ctrl), x_env_scaled, gp, y_target)

    # Bounds for gap and pressure
    bounds = [(7.0, 10.5), (0.2, 0.38)]

    # Initial guess
    x0 = np.array([9.5, 0.35])

    res = minimize(neg_ei, x0=x0, bounds=bounds, method='L-BFGS-B')

    return res.x  # Optimal (gap, pressure)

# 5. Bayesian Optimization for the second coil
BO_iter_second = 10
bo_log_second = []
gp = GaussianProcess(kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2), noise=1e-2)

for t in range(initial_obs_count, initial_obs_count + BO_iter_second):
    x_env_t_second = second_coil_df.iloc[t - 1][env_features].values.astype(np.float64)
    y_target_t_second = second_coil_df.iloc[t]["target"]

    gp.fit(X_train_second, y_train_second)
    gp.optimize_hyperparameters(bounds=((1e-2, 10.0), (1e-2, 10.0)))

    x_next_env_scaled_second = scaler_env.transform(x_env_t_second.reshape(1, -1))[0]
    x_next_ctrl_second = suggest_next_LBFGSB(gp, x_env_t_second, y_target_t_second)
    x_next_full_second = np.concatenate([x_next_env_scaled_second, x_next_ctrl_second])
    y_next_second, _ = gp.predict(x_next_full_second.reshape(1, -1))
    y_next_second = y_next_second[0]  # Ensure correct shape

    # Record results
    bo_log_second.append({
        "iteration": t,
        "n_train": len(X_train_second),
        "gap": x_next_ctrl_second[0],
        "pressure": x_next_ctrl_second[1],
        "predicted_weight": y_next_second,
        "target": y_target_t_second,
        "loss": abs(y_next_second - y_target_t_second)
    })

    # Add new data
    X_train_second = np.vstack([X_train_second, x_next_full_second])
    y_train_second = np.append(y_train_second, y_next_second)

# 6. Results Visualization for Second Coil
bo_df_second = pd.DataFrame(bo_log_second)
print(bo_df_second)

# Plotting gap, pressure, and coating weight
plt.figure(figsize=(12, 6))

# Gap and Pressure
plt.subplot(1, 2, 1)
plt.scatter(second_coil_df["gap_top"], second_coil_df["pressure_top"], 
            color="gray", alpha=0.5, label=f"{second_coil_name} (true data)")
colors = plt.cm.plasma(np.linspace(0, 1, len(bo_df_second)))
for i, row in bo_df_second.iterrows():
    plt.scatter(row["gap"], row["pressure"], color=colors[i], label=f"BO iter {int(row['iteration'])}")
plt.xlabel("Gap")
plt.ylabel("Pressure")
plt.title("BO Suggested (Gap, Pressure) vs True Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Coating Weight
plt.subplot(1, 2, 2)
plt.plot(bo_df_second["iteration"], bo_df_second["predicted_weight"], label="Predicted Weight", marker='o')
plt.plot(bo_df_second["iteration"], bo_df_second["target"], label="Target Weight", marker='x')
plt.xlabel("Iteration")
plt.ylabel("Coating Weight")
plt.title("Predicted vs Target Coating Weight")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

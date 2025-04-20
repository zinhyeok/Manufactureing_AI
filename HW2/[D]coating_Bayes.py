
import pandas as pd
import numpy as np
from gp import GaussianProcess
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("coating_sampled.csv")

# Feature definitions
features_top = ['thickness', 'width', 'speed', 'tension', 'gap_top', 'pressure_top', 'angle_top']
env_features = ['thickness', 'width', 'speed', 'tension', 'angle_top']
control_features = ['gap_top', 'pressure_top']
target_col = 'weight_top'

# Detect coil transitions
coil_changes = df['coil'].astype(str).str.strip()
change_points = coil_changes.ne(coil_changes.shift()).cumsum()
coil_transitions = coil_changes.groupby(change_points).first().reset_index(drop=True)
transition_indices = coil_changes.ne(coil_changes.shift()).to_numpy().nonzero()[0]

# Expected Improvement acquisition function
def expected_improvement(X_candidate, gp_model, y_best):
    mu, cov = gp_model.predict(X_candidate)
    sigma = np.sqrt(np.diag(cov))
    sigma = np.maximum(sigma, 1e-6)
    Z = (y_best - mu) / sigma
    ei = (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# BO results
bo_results = []

# Iterate over each transition
for i in range(1, len(transition_indices)):
    prev_start = transition_indices[i-1]
    prev_end = transition_indices[i]
    new_start = transition_indices[i]
    new_end = transition_indices[i+1] if i+1 < len(transition_indices) else len(df)

    prev_data = df.iloc[prev_start:prev_end]
    new_data = df.iloc[new_start:new_end]
    init_data = new_data.head(5)

    # GP training data
    X_env_prev = prev_data[env_features].values
    X_ctrl_prev = prev_data[control_features].values
    y_prev = prev_data[target_col].values

    scaler = StandardScaler()
    X_env_scaled_prev = scaler.fit_transform(X_env_prev)
    X_train = np.hstack([X_env_scaled_prev, X_ctrl_prev])
    y_train = y_prev

    # Train GP
    gp = GaussianProcess(kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2), noise=1e-2)
    gp.fit(X_train, y_train)
    gp.optimize_hyperparameters(bounds=((1e-3, 100.0), (1e-3, 100.0)))

    # Use fixed env value from latest of init_data
    X_env_fixed = scaler.transform(init_data[env_features].values)[-1]

    for t in range(5):
        def acq_neg(x):
            x = np.array(x).reshape(1, -1)
            x_full = np.hstack([X_env_fixed.reshape(1, -1), x])
            return -expected_improvement(x_full, gp, y_train.min())[0]

        res = minimize(acq_neg, x0=np.array([9.0, 0.27]), bounds=[(5, 13), (0.2, 0.35)])
        best_x = res.x
        x_new_full = np.hstack([X_env_fixed, best_x])
        y_new = gp.predict(x_new_full.reshape(1, -1))[0][0]

        # Update model
        X_train = np.vstack([X_train, x_new_full])
        y_train = np.append(y_train, y_new)
        gp.fit(X_train, y_train)

        # Record
        bo_results.append({
            'coil': coil_transitions[i],
            'iter': t,
            'gap_top': best_x[0],
            'pressure_top': best_x[1],
            'predicted_weight': y_new
        })

# visualize results

bo_df = pd.DataFrame(bo_results)

for coil in bo_df['coil'].unique():
    subset = bo_df[bo_df['coil'] == coil]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # GAP
    axs[0].plot(subset['iter'], subset["gap_top"], label="BO Gap", marker='x')
    axs[0].set_title(f"Gap (Top-side) - {coil}")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Gap Value")
    axs[0].legend()
    axs[0].grid(True)

    # PRESSURE
    axs[1].plot(subset['iter'], subset["pressure_top"], label="BO Pressure", marker='x')
    axs[1].set_title(f"Pressure (Top-side) - {coil}")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Pressure Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(f"Top-side BO Control Variables - Coil {coil}", fontsize=14)
    plt.tight_layout()
    plt.show()
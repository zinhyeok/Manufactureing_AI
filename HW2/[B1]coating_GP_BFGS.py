from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gp import GaussianProcess

# Load the dataset
df = pd.read_csv("coating_sampled.csv")
#data 구성
#input: thickness, width, speed, tension, gap, pressure, angle
#conntrol variable: gap, pressure
#ouput: weight
#steel has two sides: top and bottom -> 2 outputs and model neeeds, and gap, pressure, angle hasw top, bot
# Step1: weight prediction using input variables
# Step2: find optimal control variable which min diff of weight, assume that gap and pressure values are unknown?

# === Top-side prediction and optimization ===
# Step 1: Predicting weight using input variables
# === Top-side GP prediction and optimization ===

#데이터 분할 (Top-side 모델 학습용)
features_top = ['thickness', 'width', 'speed', 'tension', 'gap_top', 'pressure_top']
env_features_top = ['thickness', 'width', 'speed', 'tension']
ctrl_features_top = ['gap_top', 'pressure_top']
target_top = 'weight_top'

# 전체 feature를 분리
X_env = df[env_features_top].values
X_ctrl = df[ctrl_features_top].values
y = df[target_top].values

# Train/test split
X_env_train, X_env_test, X_ctrl_train, X_ctrl_test, y_train, y_test = train_test_split(
    X_env, X_ctrl, y, test_size=0.2, random_state=42
)

# 환경 변수만 스케일링
scaler_env = StandardScaler()
X_env_train_scaled = scaler_env.fit_transform(X_env_train)
X_env_test_scaled = scaler_env.transform(X_env_test)

# 전체 입력 조합
X_train = np.hstack([X_env_train_scaled, X_ctrl_train])
X_test = np.hstack([X_env_test_scaled, X_ctrl_test])

# GP 모델 학습
gp = GaussianProcess(kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2), noise=1e-2)
gp.fit(X_train, y_train)
gp.optimize_hyperparameters(bounds=((1e-3, 100.0), (1e-3, 100.0)))  # Optimize hyperparameters + refit

# Step 3: 예측 및 평가
y_pred, cov = gp.predict(X_test)
std = np.sqrt(np.diag(cov))

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 시각화
plt.figure(figsize=(6, 6))
plt.errorbar(y_test, y_pred, yerr=2 * std, fmt='o', alpha=0.6, label='Prediction ± 2σ')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect prediction')
plt.xlabel("Actual weight_top")
plt.ylabel("Predicted weight_top (GP)")
plt.title(f"Prediction vs Actual (Top) | R² = {r2:.3f}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

r2, mse

# Step 2: Finding optimal control variables (gap, pressure) for top-side model

# 1. coil == CRG2188 앞 16개 추출
subset_top = df[df['coil'].str.strip() == 'CRG2188'].head(16)

# 환경 변수만 정규화
env_values_top = scaler_env.transform(subset_top[env_features_top].values)

# 목표값 및 실제 제어 변수
target_weights_top = subset_top['target'].values
true_gaps_top = subset_top['gap_top'].values
true_pressures_top = subset_top['pressure_top'].values

# 최적화 루프
optimized_gaps_top = []
optimized_pressures_top = []
predicted_weights_top = []

for i in range(len(subset_top)):
    env_input = env_values_top[i]
    target = target_weights_top[i]

    def objective(x):
        full_input = np.concatenate([env_input, x])
        mu, _ = gp.predict(full_input.reshape(1, -1))
        pred = mu[0]
        error = pred - target
        penalty = np.exp(5 * abs(error)) if error < 0 else 0
        return error**2 + penalty

    x0 = np.array([9.5, 0.35])
    bounds = [(7.0, 10.5), (0.2, 0.38)]  # 실험 기반 범위
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    x_opt = result.x
    optimized_gaps_top.append(x_opt[0])
    optimized_pressures_top.append(x_opt[1])

    mu_pred, _ = gp.predict(np.concatenate([env_input, x_opt]).reshape(1, -1))
    predicted_weights_top.append(mu_pred[0])

# 결과 정리
opt_df_top = pd.DataFrame({
    'target_weight': target_weights_top,
    'predicted_weight': predicted_weights_top,
    'true_gap': true_gaps_top,
    'opt_gap': optimized_gaps_top,
    'true_pressure': true_pressures_top,
    'opt_pressure': optimized_pressures_top,
    'weight_top': subset_top['weight_top'].values
})


# true_gap vs true_pressure & opt_gap vs opt_pressure 시각화
fig, axs = plt.subplots(1, 3, figsize=(12, 5))

# GAP
axs[0].plot(opt_df_top.index, opt_df_top["true_gap"], label="True Gap", marker='o')
axs[0].plot(opt_df_top.index, opt_df_top["opt_gap"], label="Optimized Gap", marker='x')
axs[0].set_title("Gap (Top-side)")
axs[0].set_xlabel("Sample Index")
axs[0].set_ylabel("Gap Value")
axs[0].legend()
axs[0].grid(True)

# PRESSURE
axs[1].plot(opt_df_top.index, opt_df_top["true_pressure"], label="True Pressure", marker='o')
axs[1].plot(opt_df_top.index, opt_df_top["opt_pressure"], label="Optimized Pressure", marker='x')
axs[1].set_title("Pressure (Top-side)")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Pressure Value")
axs[1].legend()
axs[1].grid(True)

# WEIGHT
axs[2].plot(opt_df_top.index, opt_df_top["target_weight"], label="Target Weight", marker='o')
axs[2].plot(opt_df_top.index, opt_df_top["predicted_weight"], label="Predicted Weight", marker='x')
axs[2].plot(opt_df_top.index, opt_df_top["weight_top"], label="Actual Weight", marker='^')
axs[2].get_yaxis().get_major_formatter().set_useOffset(False)
axs[2].set_title("Weight Prediction (Top-side)")
axs[2].set_xlabel("Sample Index")
axs[2].set_ylabel("Weight Value")
axs[2].legend()
axs[2].grid(True)

plt.suptitle("Top-side: True vs Optimized Control Variables", fontsize=14)
plt.tight_layout()
plt.show()

# === Bottom-side GP prediction and optimization ===

# Step 1: Predicting weight using input variables (Bottom-side model)

# 1. 변수 설정
features_bot = ['thickness', 'width', 'speed', 'tension', 'gap_bot', 'pressure_bot']
env_features_bot = ['thickness', 'width', 'speed', 'tension']
ctrl_features_bot = ['gap_bot', 'pressure_bot']
target_bot = 'weight_bot'

# 2. 데이터 분할 및 환경 변수만 정규화
X_env_bot = df[env_features_bot].values
X_ctrl_bot = df[ctrl_features_bot].values
yb = df[target_bot].values

X_env_train, X_env_test, X_ctrl_train, X_ctrl_test, yb_train, yb_test = train_test_split(
    X_env_bot, X_ctrl_bot, yb, test_size=0.2, random_state=42
)

scaler_env_bot = StandardScaler()
X_env_train_scaled = scaler_env_bot.fit_transform(X_env_train)
X_env_test_scaled = scaler_env_bot.transform(X_env_test)

Xb_train = np.hstack([X_env_train_scaled, X_ctrl_train])
Xb_test = np.hstack([X_env_test_scaled, X_ctrl_test])

# ✅ Gaussian Process 모델 학습
gp_bot = GaussianProcess(
    kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2),
    noise=1e-2
)
gp_bot.fit(Xb_train, yb_train)
gp_bot.optimize_hyperparameters(bounds=((1e-3, 100.0), (1e-3, 100.0)))  # Optimize hyperparameters + refit

# 예측 및 평가
yb_pred, cov_bot = gp_bot.predict(Xb_test)
std_bot = np.sqrt(np.diag(cov_bot))

r2 = r2_score(yb_test, yb_pred)
mse = mean_squared_error(yb_test, yb_pred)

# 시각화
plt.figure(figsize=(6, 6))
plt.errorbar(yb_test, yb_pred, yerr=2 * std_bot, fmt='o', alpha=0.6, label='Prediction ± 2σ')
plt.plot([min(yb_test), max(yb_test)], [min(yb_test), max(yb_test)], color='red', linestyle='--', label='Perfect prediction')
plt.xlabel("Actual weight_bot")
plt.ylabel("Predicted weight_bot (GP)")
plt.title(f"Prediction vs Actual (bot) | R² = {r2:.3f}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

r2, mse

# Step 2: Finding optimal control variables (gap, pressure) for bottom-side model

# 1. coil == CRG2188 앞 16개 추출 (bot용)
subset_bot = df[df['coil'].str.strip() == 'CRG2188'].head(16)

# 환경 변수만 정규화
env_values_bot = scaler_env_bot.transform(subset_bot[env_features_bot].values)

# 목표값 및 실제 제어 변수
target_weights_bot = subset_bot['target'].values
true_gaps_bot = subset_bot['gap_bot'].values
true_pressures_bot = subset_bot['pressure_bot'].values

# BFGS 최적화 루프
optimized_gaps_bot = []
optimized_pressures_bot = []
predicted_weights_bot = []

for i in range(len(subset_bot)):
    env_input = env_values_bot[i]
    target = target_weights_bot[i]

    def objective(x):
        full_input = np.concatenate([env_input, x])  # [정규화된 환경 변수, 원 단위 제어 변수]
        mu, _ = gp_bot.predict(full_input.reshape(1, -1))
        pred = mu[0]
        penalty = np.exp(5 * (target - pred)) if pred < target else 0
        return (pred - target)**2 + penalty

    x0 = np.array([9.5, 0.35])
    bounds = [(7.0, 10.5), (0.2, 0.38)]  # gap, pressure의 실험 기반 범위
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    x_opt = result.x
    optimized_gaps_bot.append(x_opt[0])
    optimized_pressures_bot.append(x_opt[1])

    mu_pred, _ = gp_bot.predict(np.concatenate([env_input, x_opt]).reshape(1, -1))
    predicted_weights_bot.append(mu_pred[0])

# 결과 정리
opt_df_bot = pd.DataFrame({
    'target_weight': target_weights_bot,
    'predicted_weight': predicted_weights_bot,
    'true_gap': true_gaps_bot,
    'opt_gap': optimized_gaps_bot,
    'true_pressure': true_pressures_bot,
    'opt_pressure': optimized_pressures_bot,
    'weight_bot': subset_bot['weight_bot'].values
})

# true_gap vs true_pressure & opt_gap vs opt_pressure 시각화
fig, axs = plt.subplots(1, 3, figsize=(12, 5))

# GAP
axs[0].plot(opt_df_bot.index, opt_df_bot["true_gap"], label="True Gap", marker='o')
axs[0].plot(opt_df_bot.index, opt_df_bot["opt_gap"], label="Optimized Gap", marker='x')
axs[0].set_title("Gap (Bottom-side)")
axs[0].set_xlabel("Sample Index")
axs[0].set_ylabel("Gap Value")
axs[0].legend()
axs[0].grid(True)

# PRESSURE
axs[1].plot(opt_df_bot.index, opt_df_bot["true_pressure"], label="True Pressure", marker='o')
axs[1].plot(opt_df_bot.index, opt_df_bot["opt_pressure"], label="Optimized Pressure", marker='x')
axs[1].set_title("Pressure (Bottom-side)")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Pressure Value")
axs[1].legend()
axs[1].grid(True)

# WEIGHT
axs[2].plot(opt_df_bot.index, opt_df_bot["target_weight"], label="Target Weight", marker='o')
axs[2].plot(opt_df_bot.index, opt_df_bot["predicted_weight"], label="Predicted Weight", marker='x')
axs[2].plot(opt_df_bot.index, opt_df_bot["weight_bot"], label="Actual Weight", marker='^')
axs[2].get_yaxis().get_major_formatter().set_useOffset(False)
axs[2].set_title("Weight Prediction (Bottom-side)")
axs[2].set_xlabel("Sample Index")
axs[2].set_ylabel("Weight Value")
axs[2].legend()
axs[2].grid(True)


plt.suptitle("Bottom-side: True vs Optimized Control Variables", fontsize=14)
plt.tight_layout()
plt.show()
import pandas as pd

# Load the dataset
df = pd.read_csv("coating_sampled.csv")
#data 구성
#input: thickness, width, speed, tension, gap, pressure, angle
#conntrol variable: gap, pressure
#ouput: weight
#steel has two sides: top and bottom -> 2 outputs and model neeeds, and gap, pressure, angle hasw top, bot
# Step1: weight prediction using input variables
# Step2: find optimal control variable which min diff of weight, assume that gap and pressure values are unknown??


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Top-side prediction and optimization ===
# Step 1: Predicting weight using input variables
# 1. 데이터 분할 (Top-side 모델 학습용)
features_top = ['thickness', 'width', 'speed', 'tension', 'gap_top', 'pressure_top', 'angle_top']
target_top = 'weight_top'

X = df[features_top].values
y = df[target_top].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 데이터 전처리 (StandardScaler 사용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP 모델 학습
mlp_top = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000, random_state=42)
mlp_top.fit(X_train, y_train)

# 3. 예측 및 평가
y_pred = mlp_top.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 시각화용 데이터 정리
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual weight_top")
plt.ylabel("Predicted weight_top")
plt.title(f"Prediction vs Actual (Top) | R² = {r2:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()

r2, mse

#Step 2: Finding optimal control variables (gap, pressure) for top-side model
# 1. coil == CRG2188 앞 16개 추출 (bot용)
subset_top = df[df['coil'].str.strip() == 'CRG2188'].head(16)

# 환경 변수 추출 및 정규화 (top 기준)
env_features_top = ['thickness', 'width', 'speed', 'tension', 'angle_top']
env_values_top = scaler.transform(subset_top[features_top].values)[:, :len(env_features_top)]

# 목표값 및 실제 제어 변수
target_weights_top = subset_top['weight_top'].values
true_gaps_top = subset_top['gap_top'].values
true_pressures_top = subset_top['pressure_top'].values

# BFGS 최적화 루프
optimized_gaps_top = []
optimized_pressures_top = []
predicted_weights_top = []

for i in range(len(subset_top)):
    env_input = env_values_top[i]
    target = target_weights_top[i]

    def objective(x):
        full_input_raw = np.concatenate([env_input, x])
        pred = mlp_top.predict(full_input_raw.reshape(1, -1))
        return (pred[0] - target) ** 2

    x0 = np.array([9.0, 0.27])
    result = minimize(objective, x0, method='BFGS')

    optimized_gaps_top.append(result.x[0])
    optimized_pressures_top.append(result.x[1])
    predicted_weights_top.append(mlp_top.predict(np.concatenate([env_input, result.x]).reshape(1, -1))[0])

# 결과 정리
opt_df_top = pd.DataFrame({
    'target_weight': target_weights_top,
    'predicted_weight': predicted_weights_top,
    'true_gap': true_gaps_top,
    'opt_gap': optimized_gaps_top,
    'true_pressure': true_pressures_top,
    'opt_pressure': optimized_pressures_top
})

# true_gap vs true_pressure & opt_gap vs opt_pressure 시각화
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

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

plt.suptitle("Top-side: True vs Optimized Control Variables", fontsize=14)
plt.tight_layout()
plt.show()

# === Bottom-side prediction and optimization ===

# Step 1: Predicting weight using input variables (Bottom-side model)
# 1. 변수 설정
features_bot = ['thickness', 'width', 'speed', 'tension', 'gap_bot', 'pressure_bot', 'angle_bot']
target_bot = 'weight_bot'

Xb = df[features_bot].values
yb = df[target_bot].values

# 2. 정규화 및 분할
scaler_bot = StandardScaler()
Xb_scaled = scaler_bot.fit_transform(Xb)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb_scaled, yb, test_size=0.2, random_state=42)

# 3. MLP 모델 학습
mlp_bot = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000, random_state=42)
mlp_bot.fit(Xb_train, yb_train)

yb_pred = mlp_bot.predict(Xb_test)
r2 = r2_score(yb_test, yb_pred)
mse = mean_squared_error(yb_test, yb_pred)

# 시각화용 데이터 정리
plt.figure(figsize=(6, 6))
plt.scatter(yb_test, yb_pred, alpha=0.7)
plt.plot([min(yb_test), max(yb_test)], [min(yb_train), max(yb_test)], color='red', linestyle='--')
plt.xlabel("Actual weight_bot")
plt.ylabel("Predicted weight_bot")
plt.title(f"Prediction vs Actual (Bot) | R² = {r2:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()

r2, mse

# Step 2: Finding optimal control variables (gap, pressure) for bottom-side model
# 1. coil == CRG2188 앞 16개 추출 (bot용)
subset_bot = df[df['coil'].str.strip() == 'CRG2188'].head(16)

# 환경 변수 추출 및 정규화 (bot 기준)
env_features_bot = ['thickness', 'width', 'speed', 'tension', 'angle_bot']
env_values_bot = scaler_bot.transform(subset_bot[features_bot].values)[:, :len(env_features_bot)]

# 목표값 및 실제 제어 변수
target_weights_bot = subset_bot['weight_bot'].values
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
        full_input_raw = np.concatenate([env_input, x])
        pred = mlp_bot.predict(full_input_raw.reshape(1, -1))
        return (pred[0] - target) ** 2

    x0 = np.array([9.0, 0.27])
    result = minimize(objective, x0, method='BFGS')

    optimized_gaps_bot.append(result.x[0])
    optimized_pressures_bot.append(result.x[1])
    predicted_weights_bot.append(mlp_bot.predict(np.concatenate([env_input, result.x]).reshape(1, -1))[0])

# 결과 정리
opt_df_bot = pd.DataFrame({
    'target_weight': target_weights_bot,
    'predicted_weight': predicted_weights_bot,
    'true_gap': true_gaps_bot,
    'opt_gap': optimized_gaps_bot,
    'true_pressure': true_pressures_bot,
    'opt_pressure': optimized_pressures_bot
})

# true_gap vs true_pressure & opt_gap vs opt_pressure 시각화
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# GAP
axs[0].plot(opt_df_bot.index, opt_df_bot["true_gap"], label="True Gap", marker='o')
axs[0].plot(opt_df_bot.index, opt_df_bot["opt_gap"], label="Optimized Gap", marker='x')
axs[0].set_title("Gap (Buttom-side)")
axs[0].set_xlabel("Sample Index")
axs[0].set_ylabel("Gap Value")
axs[0].legend()
axs[0].grid(True)

# PRESSURE
axs[1].plot(opt_df_bot.index, opt_df_bot["true_pressure"], label="True Pressure", marker='o')
axs[1].plot(opt_df_bot.index, opt_df_bot["opt_pressure"], label="Optimized Pressure", marker='x')
axs[1].set_title("Pressure (Buttom-side)")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Pressure Value")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Buttom-side: True vs Optimized Control Variables", fontsize=14)
plt.tight_layout()
plt.show()
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GaussianProcess:
    def __init__(self, kernel, noise=1e-3):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))

    def predict(self, X_s):
        X_s = np.array(X_s)
        K_s = self.kernel(self.X_train, X_s)
        K_ss = self.kernel(X_s, X_s) + self.noise * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T @ K_inv @ self.y_train
        cov_s = K_ss - K_s.T @ K_inv @ K_s

        return mu_s, cov_s

    def log_marginal_likelihood(self, params):
        l, sigma_f = params
        self.kernel = lambda X1, X2: self.squared_exponential_kernel(X1, X2, l, sigma_f)
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_inv = np.linalg.inv(K)

        term1 = -0.5 * self.y_train.T @ K_inv @ self.y_train
        # term2 = -0.5 * np.log(np.linalg.det(K))
        L = np.linalg.cholesky(K)
        term2 = -np.sum(np.log(np.diagonal(L))) * 2
        term3 = -0.5 * len(self.y_train) * np.log(2 * np.pi)

        return -(term1 + term2 + term3)

#kernel hyperparameters optimization
    def optimize_hyperparameters(self, bounds):
        res = minimize(
            fun=self.log_marginal_likelihood,
            x0=[0.5, 0.5],
            bounds=bounds,
            method='L-BFGS-B'
        )
        l_opt, sigma_f_opt = res.x
        self.kernel = lambda X1, X2: self.squared_exponential_kernel(X1, X2, l_opt, sigma_f_opt)
        self.fit(self.X_train, self.y_train) # Refit with optimized hyperparameters

    @staticmethod
    def squared_exponential_kernel(X1, X2, l=1.0, sigma_f=1.0):
        """Squared Exponential Kernel (RBF Kernel)"""
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        result = sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
        return result

# Example usage
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("goldstein_price.csv")
    X_train = data[['X1', 'X2']].values
    y_train = data['Y'].values

    # Define test points for surface prediction
    # Create a grid of points for prediction(for 0-1 range in 2D similar to the train data)
    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.c_[X1.ravel(), X2.ravel()]

    # Initialize and train GP
    gp = GaussianProcess(kernel=lambda X1, X2: GaussianProcess.squared_exponential_kernel(X1, X2))
    gp.fit(X_train, y_train)
    gp.optimize_hyperparameters(bounds=((1e-2, 10.0), (1e-2, 10.0))) # Optimize hyperparameters + refit
    
    # Make predictions
    mu, cov = gp.predict(X_test)
    mu = mu.reshape(X1.shape)

    # Plot both: 3D scatter (left) + surface plot (right)
    fig = plt.figure(figsize=(12, 5))

    # Left: Observed data points
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, color='dodgerblue', s=40, alpha=0.7)
    ax1.set_title("Observed Data")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    # Right: GP prediction surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X1, X2, mu, cmap='Greys', edgecolor='k', alpha=0.9)
    ax2.set_title("GP Prediction Surface")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Predicted y")

    plt.tight_layout()
    plt.show()

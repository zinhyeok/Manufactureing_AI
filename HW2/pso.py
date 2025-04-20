
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# PSO 클래스 정의
class PSO:
    def __init__(self, objective_function, bounds, num_particles=30, max_iter=100, use_adaptive=True, random_seed=42):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.use_adaptive = use_adaptive
        self.particles = [self.Particle(bounds, objective_function) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].best_position.copy()
        self.global_best_value = self.particles[0].best_value
        self.history = [] # Store the best value at each iteration
        self.position_history = []  # positions of all particles per iteration
        np.random.seed(random_seed)  # Set random seed for reproducibility
    class Particle:
        def __init__(self, bounds, objective_function):
            self.position = np.random.uniform(bounds[0], bounds[1], 2)
            self.velocity = np.random.uniform(-1, 1, 2)
            self.best_position = self.position.copy()
            self.best_value = objective_function(*self.position)

        def evaluate(self, objective_function):
            value = objective_function(*self.position)
            if value < self.best_value:
                self.best_value = value
                self.best_position = self.position.copy()

        def update_velocity(self, global_best, w, c1, c2):
            r1, r2 = np.random.rand(2)
            cognitive = c1 * r1 * (self.best_position - self.position)
            social = c2 * r2 * (global_best - self.position)
            self.velocity = w * self.velocity + cognitive + social

        def update_position(self, bounds):
            self.position += self.velocity
            self.position = np.clip(self.position, bounds[0], bounds[1])

    def optimize(self):
        for t in range(self.max_iter):
            # Rule-of-thumb based inertia/coefficients
            if self.use_adaptive:
                N = self.max_iter
                w = 0.4 * ((t - N) / N**2) + 0.4
                c1 = -3 * (t / N) + 3.5
                c2 = 3 * (t / N) + 0.5
            else:
                w, c1, c2 = 0.7, 1.5, 1.5  # Fixed values

            for p in self.particles:
                p.evaluate(self.objective_function)
                if p.best_value < self.global_best_value:
                    self.global_best_position = p.best_position.copy()
                    self.global_best_value = p.best_value

            for p in self.particles:
                p.update_velocity(self.global_best_position, w, c1, c2)
                p.update_position(self.bounds)

            self.history.append(self.global_best_value)
            self.position_history.append([p.position.copy() for p in self.particles])

        return self.global_best_position, self.global_best_value, self.history, self.position_history

# Example usage
if __name__ == "__main__":
    # Example objective function
    def objective_function(x, y):
        return x**2 + (y+1)**2 + 5*np.cos(1.5*x + 1.5) -3*np.cos(2*y - 1.5) 

    # Visualize PSO optimization process
    def visualize_pso(position_history, bounds=[-10, 10], interval=10, objective_function=objective_function):
        x = np.linspace(bounds[0], bounds[1], 200)
        y = np.linspace(bounds[0], bounds[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = objective_function(X, Y)

        steps = list(range(0, len(position_history), interval))
        num_rows = len(steps)

        fig = plt.figure(figsize=(18, 8 * num_rows))

        for i, idx in enumerate(steps):
            positions = np.array(position_history[idx])

            # 2D contour plot
            ax2d = fig.add_subplot(num_rows, 2, 2 * i + 1)
            ax2d.contourf(X, Y, Z, levels=50, cmap=cm.magma, alpha=0.7)
            ax2d.scatter(positions[:, 0], positions[:, 1], c='black', s=20)
            ax2d.set_title(f"Iteration {idx}")
            ax2d.set_xlim(bounds)
            ax2d.set_ylim(bounds)
            ax2d.set_aspect('equal')

            # 3D surface plot
            ax3d = fig.add_subplot(num_rows, 2, 2 * i + 2, projection='3d')
            ax3d.plot_surface(X, Y, Z, cmap=cm.magma, alpha=0.6, edgecolor='none')
            ax3d.scatter(positions[:, 0], positions[:, 1],
                        objective_function(positions[:, 0], positions[:, 1]),
                        c='black', s=20)
            ax3d.set_title(f"Iteration {idx}")
            ax3d.set_xlim(bounds)
            ax3d.set_ylim(bounds)
            ax3d.set_zlim(np.min(Z), np.max(Z))

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.show()


    # Run visualization
    pso = PSO(objective_function, bounds=[-10, 10], max_iter=30, use_adaptive=True)
    best_pos, best_val, history, pos_history = pso.optimize()
    visualize_pso(pos_history, bounds=[-10, 10], interval=5, objective_function=objective_function)

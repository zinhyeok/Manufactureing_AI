
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# PSO 클래스 정의
class PSO:
    def __init__(self, objective_function, bounds, num_particles=30, max_iter=100, use_adaptive=True):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.use_adaptive = use_adaptive
        self.particles = [self.Particle(bounds, objective_function) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].best_position.copy()
        self.global_best_value = self.particles[0].best_value
        self.history = []

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

        return self.global_best_position, self.global_best_value, self.history

# Example objective function
def objective_function(x, y):
    return x**2 + y**2 + 10 * np.sin(x) * np.cos(y)

# Visualize PSO optimization process
def visualize_pso(objective_function, bounds=[-10, 10], num_particles=30, max_iter=50):
    # Create a mesh grid for the objective function
    x = np.linspace(bounds[0], bounds[1], 300)
    y = np.linspace(bounds[0], bounds[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    # Initialize PSO
    pso = PSO(objective_function, bounds, num_particles, max_iter)
    position_history = []

    for t in range(max_iter):
        positions = np.array([p.position for p in pso.particles])
        position_history.append(positions)
        pso.optimize()

    # Visualization steps
    selected_steps = [0, 1, 5, 10, 25, max_iter - 1]
    fig = plt.figure(figsize=(24, 10))

    for i, step in enumerate(selected_steps):
        col = i * 2  # Two plots per iteration

        # 2D contour plot
        ax1 = fig.add_subplot(2, len(selected_steps), col + 1)
        contour = ax1.contourf(X, Y, Z, levels=50, cmap='PuBuGn')  # Updated colormap
        if step < len(position_history):
            pos = position_history[step]
            ax1.scatter(pos[:, 0], pos[:, 1], color='black', s=10)
        ax1.set_title(f"2D: Iter {step}")
        ax1.set_xlim(bounds)
        ax1.set_ylim(bounds)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # 3D surface plot
        ax2 = fig.add_subplot(2, len(selected_steps), col + 2, projection='3d')
        ax2.plot_surface(X, Y, Z, cmap='PuBuGn', alpha=0.8)  # Updated colormap
        ax2.contour(X, Y, Z, levels=50, cmap='PuBuGn', offset=np.min(Z), zdir='z')  # Same colormap for contours
        if step < len(position_history):
            pos = position_history[step]
            z_pos = objective_function(pos[:, 0], pos[:, 1])
            ax2.scatter(pos[:, 0], pos[:, 1], z_pos, color='black', s=10)
        ax2.set_title(f"3D: Iter {step}")
        ax2.set_xlim(bounds)
        ax2.set_ylim(bounds)
        ax2.set_zlim(np.min(Z), np.max(Z))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('Objective')

    plt.tight_layout()
    plt.show()


# Run visualization
visualize_pso(objective_function, bounds=[-10, 10], max_iter=50)

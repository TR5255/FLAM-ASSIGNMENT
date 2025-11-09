import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed # For parallel processing
import time

# Start Timer
start_time = time.time()

# Load Data
try:
    data = pd.read_csv("xy_data.csv")
    x_data = data["x"].values
    y_data = data["y"].values
    # Pre-calculate data points array for cdist (N_data, 2)
    data_points = np.stack([x_data, y_data], axis=1)
    print(f"Loaded {len(x_data)} data points.")
except FileNotFoundError:
    print("Error: xy_data.csv not found.")
    print("Please create this file with 'x' and 'y' columns.")
    exit()


# Given Parametric Equations
def equations(t, theta, M, X):
    """Given parametric equations."""
    x = t*np.cos(theta) - np.exp(M*t)*np.sin(0.3*t)*np.sin(theta) + X
    y = 42 + t*np.sin(theta) + np.exp(M*np.abs(t))*np.sin(0.3*t)*np.cos(theta)
    return x, y


# Cost Function Optimized
def ssqfn(params):
    """
    Calculates sum of squared distances (residuals) using efficient cdist.
    """
    theta, M, X = params
    
    # Calculate curve points on the grid (N_grid, 2)
    x_curve, y_curve = equations(t_grid, theta, M, X)
    curve_points = np.stack([x_curve, y_curve], axis=1)
    
    # Use cdist to find the squared Euclidean distance
    # from each data_point to each curve_point.
    # Shape: (N_data, N_grid)
    dist_matrix_sq = cdist(data_points, curve_points, 'sqeuclidean')
    
    # Find the minimum squared distance for each data point
    min_dist2 = np.min(dist_matrix_sq, axis=1)
    
    return np.sum(min_dist2)


# t-Refinement Function
def refine_t(xi, yi, theta, M, X, t_guess):
    """
    Refines the t_i for a single point (xi, yi) given a t_guess.
    This function is called by the parallel loop.
    """
    def obj(t):
        x_pred, y_pred = equations(t, theta, M, X)
        return (x_pred - xi)**2 + (y_pred - yi)**2
        
    # Bound the search space around the t_guess
    lower = max(6, t_guess - 1.0)
    upper = min(60, t_guess + 1.0)
    
    res = minimize_scalar(obj, bounds=(lower, upper), method="bounded")
    return res.x, res.fun


# Main Execution
# Generate a grid of t values
t_grid = np.linspace(6, 60, 800)

# Given bounds on the parameters
bounds = [
    (np.deg2rad(0.1), np.deg2rad(50)),  # theta in radians
    (-0.05, 0.05),                      # M
    (0, 100)                            # X
]

# 1. Global Optimization
print("Starting global optimization (L-BFGS-B)...")
opt_start_time = time.time()
parameters = minimize(ssqfn, x0=[np.deg2rad(25), 0.02, 50.0],
                      bounds=bounds, method="L-BFGS-B")
opt_end_time = time.time()
print(f"Optimization finished in {opt_end_time - opt_start_time:.2f} seconds.")

theta_opt, M_opt, X_opt = parameters.x

print("\nOptimal parameters:")
print(f"theta (deg) = {np.rad2deg(theta_opt):.4f}")
print(f"M = {M_opt:.4f}")
print(f"X = {X_opt:.4f}")


# 2. Refine t_i for each data point (Vectorized + Parallel)
print("\nRefining t_i for each data point...")
refine_start_time = time.time()

# 2a. Precompute optimal curve and find *all* t_guesses (vectorized)
x_curve_opt, y_curve_opt = equations(t_grid, theta_opt, M_opt, X_opt)
curve_points_opt = np.stack([x_curve_opt, y_curve_opt], axis=1)

dist_matrix_opt = cdist(data_points, curve_points_opt, 'sqeuclidean')
nearest_grid_indices = np.argmin(dist_matrix_opt, axis=1)
t_guesses = t_grid[nearest_grid_indices]

# 2b. Run refinement in parallel (using all available cores: n_jobs=-1)
# (refine_t function was defined earlier)
results_list = Parallel(n_jobs=-1)(
    delayed(refine_t)(x_data[i], y_data[i], theta_opt, M_opt, X_opt, t_guesses[i]) 
    for i in range(len(x_data))
)

# Unpack parallel results
t_refined = [res[0] for res in results_list]
residuals = [res[1] for res in results_list]

refine_end_time = time.time()
print(f"Refinement finished in {refine_end_time - refine_start_time:.2f} seconds.")


# 3. Save and Plot Results
results = pd.DataFrame({
    "x": x_data,
    "y": y_data,
    "t_fit": t_refined,
    "residual": residuals,
    "theta": [np.rad2deg(theta_opt)]*len(x_data),
    "M": [M_opt]*len(x_data),
    "X": [X_opt]*len(x_data)
})
results.to_csv("fitted_values.csv", index=False)
print("\nResults saved to fitted_values.csv")

# Plot
t_grid_dense = np.linspace(6, 60, 1000)
x_fit, y_fit = equations(t_grid_dense, theta_opt, M_opt, X_opt)

plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, label="Data", color="red", s=10) 
plt.plot(x_fit, y_fit, label="Fitted curve", color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Parametric Fit with Refined t_i")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

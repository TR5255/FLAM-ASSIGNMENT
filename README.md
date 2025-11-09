# FLAM-ASSIGNMENT
# Parametric Curve Fitting with t-Refinement

This repository contains a Python implementation for fitting a parametric curve to 2D data points using a two-stage approach:

1. **Global optimization** of the curve parameters (θ, M, X) by minimizing the sum of squared distances between data points and a grid-sampled curve (using `scipy.optimize.minimize`).
2. **Per-point t-refinement**: after finding optimal parameters, each data point's corresponding curve parameter `t_i` is refined using a local bounded 1D minimization (`scipy.optimize.minimize_scalar`) in parallel (via `joblib`).

The implementation is designed for speed and robustness: it uses vectorized distance computations (`scipy.spatial.distance.cdist`) for the global cost and parallelizes the per-point refinement step.

---

## Repository structure

```
/ (root)
├── xy_data.csv           # Input CSV with columns: x, y
├── FLAM_ASSIGNMENT.py     # Main script (the code you provided)
├── README.md             # This file
```

---

## Requirements

* Python 3.8+ recommended
* Key Python packages:

  * numpy
  * pandas
  * matplotlib
  * scipy
  * joblib

---

## How the code works (brief)

* **Data loading**: reads `xy_data.csv` expecting two columns named `x` and `y`.
* **Parametric curve**: implemented in `equations(t, theta, M, X)` which returns the (x,y) coordinates for a given `t` and parameters.
* **Global objective (`ssqfn`)**:

  * Samples the curve on a dense grid of `t` values (`t_grid`).
  * Computes squared Euclidean distances between every data point and every sampled curve point via `cdist` (vectorized).
  * For each data point, takes the minimum squared distance to the curve grid and sums them to form the objective.
* **Global optimization**: `scipy.optimize.minimize` (L-BFGS-B) is used to find `theta`, `M`, `X` within user-specified bounds.
* **t-refinement**:

  * For each data point, take the `t` value of the nearest grid point as a starting guess.
  * Run `minimize_scalar` in a small bounded interval around the guess to refine the `t_i` that minimizes squared distance to the curve.
  * Runs the refinement in parallel using `joblib.Parallel(n_jobs=-1)`.
* **Outputs**: `fitted_values.csv` contains per-point (x, y, t_fit, residual, theta, M, X). A matplotlib plot is also shown.

---

## Usage

1. Place your data in `xy_data.csv` with columns `x` and `y`.
2. Run the script:

```bash
python FLAM_ASSIGNMENT.py
```

The script prints progress and timing, saves `fitted_values.csv`, and displays a plot of the data and the fitted curve.

### Configuration knobs to tune

* `t_grid = np.linspace(6, 60, 800)`: sampling range and density for the global search. Increase density for more accurate global objective (at cost of memory/time).
* `bounds`: constraints for `theta`, `M`, and `X` used by the optimizer.
* Initial guess `x0=[np.deg2rad(25), 0.02, 50.0]`: change if you have domain knowledge.
* `minimize_scalar` bounding window (`t_guess +/- 1.0`): widen if points can be far from the initial grid guess.
* `n_jobs=-1` in `Parallel`: controls CPU usage; set to a specific number to limit cores.

---

## Output file description

`fitted_values.csv` contains the following columns:

* `x`, `y`: original data point coordinates.
* `t_fit`: refined parameter value `t_i` for each data point.
* `residual`: squared residual (distance^2) after refinement.
* `theta`, `M`, `X`: the optimized global parameters (repeated for each row).

---

## Performance notes & tips

* The global objective uses an `O(N_data * N_grid)` distance matrix. For very large datasets or extremely dense `t_grid`, memory may become the bottleneck. To mitigate:

  * Reduce `t_grid` density (smaller `N_grid`) and rely more on refinement.
  * Use a chunked or streaming approach to compute distances in blocks.
* The per-point refinement step is parallelized. If you run into memory contention or CPU throttling, try setting `n_jobs` to a smaller value.
* If the optimizer is slow or converges to a local minimum, try multiple random restarts (call `minimize` from several different `x0` values and pick the best result).

---

## Extensions and ideas

* **Weighted residuals**: support anisotropic or heteroscedastic noise by adding per-point weights.
* **Analytic projection**: if possible, derive and use the analytic projection of a data point onto the parametric curve instead of `minimize_scalar` to speed up refinement.
* **Robust loss**: replace least-squares with a robust loss (e.g., Huber) to reduce sensitivity to outliers.
* **Visualization**: save static plot images (`.png`) and include an interactive viewer (e.g., Plotly) for inspecting point-to-curve correspondences.

---


---

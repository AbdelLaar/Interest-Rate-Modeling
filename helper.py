import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from scipy.stats import ncx2
from numpy.lib.scimath import sqrt as csqrt 


#######
# CIR Simulation functions
#######

def cir_simulate(r0, kappa, theta, sigma, T, dt, n_paths=1000, rng=None):
    """
    Simulating the CIR using the noncentral-chi-square distribution

    dr_t = kappa (theta - r_t) dt + sigma * sqrt(r_t) dW_t

    Returns: array (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.round(T / dt))
    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = r0

    e_kdt = np.exp(-kappa * dt)
    c = (sigma**2 * (1 - e_kdt)) / (4 * kappa)
    d = 4 * kappa * theta / sigma**2  # degree of freedom

    for i in range(n_steps):
        lam = (4 * kappa * e_kdt / (sigma**2 * (1 - e_kdt))) * np.maximum(rates[:, i], 0.0)
        rates[:, i + 1] = c * rng.noncentral_chisquare(df=d, nonc=lam, size=rates.shape[0])

    return rates

def cir_mean_curve(r0, kappa, theta, t):
    """E[r_t | r0]"""
    t = np.asarray(t)
    e = np.exp(-kappa * t)
    return r0 * e + theta * (1 - e)


def cir_moments(r0, kappa, theta, sigma, t):
    e = np.exp(-kappa * t)
    mean = r0 * e + theta * (1 - e)
    var = (sigma**2 / kappa) * (1 - e) * (r0 * e + 0.5 * theta * (1 - e))
    return {"mean": float(mean), "var": float(var), "std_dev": float(np.sqrt(var))}

def cir_quantile(r0, kappa, theta, sigma, t, q):
    """quantile of r_t via noncentral-chi-square"""
    if t == 0:
        return r0
    e = np.exp(-kappa * t)
    c = (sigma**2 * (1 - e)) / (4 * kappa)
    d = 4 * kappa * theta / sigma**2
    lam = (4 * kappa * e / (sigma**2 * (1 - e))) * r0
    return float(c * ncx2.ppf(q, d, lam))

def cir_quantile_curve(r0, kappa, theta, sigma, time_grid, q):
    """Vectorized quantiles"""
    return np.array([cir_quantile(r0, kappa, theta, sigma, t, q) for t in time_grid])

def cir_bond_price_analytical(r0, kappa, theta, sigma, T):
    """P(0,T) under CIR, using the formula
    
    A(T) * exp(-B(T) r0)
    
    """
    if T <= 0: return 1.0
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    exp_gT = np.exp(gamma * T)
    B = (2 * (exp_gT - 1)) / (2 * gamma + (kappa + gamma) * (exp_gT - 1))
    A = ((2 * gamma * np.exp((kappa + gamma) * T / 2)) /
         (2 * gamma + (kappa + gamma) * (exp_gT - 1))) ** (2 * kappa * theta / sigma**2)
    return A * np.exp(-B * r0)

def cir_bond_price_monte_carlo(r0, kappa, theta, sigma, T, dt=0.01, n_paths=10000, rng=None):
    """
    P(0,T) = E[exp(-∫_0^T r_s ds)]
    """
    r_paths = cir_simulate(r0, kappa, theta, sigma, T, dt, n_paths=n_paths, rng=rng)
    time_grid = np.linspace(0.0, T, r_paths.shape[1])
    integ = np.trapezoid(r_paths, time_grid, axis=1)
    disc = np.exp(-integ)
    price = float(np.mean(disc))
    stderr = float(np.std(disc, ddof=1) / np.sqrt(n_paths))
    return price, stderr


#######
# Helper functions 
#######

def cumulative_trapz_on_grid(paths, dt):
    # average on each interval * dt
    incr = 0.5 * (paths[:, 1:] + paths[:, :-1]) * dt
    cumul = np.cumsum(incr, axis=1) 
    return np.hstack([np.zeros((paths.shape[0], 1)), cumul])

def _phi_triplet(which, kappa, theta, sigma):
    """
    We compute the phi functions
      x:  phi1 = sqrt(k^2 + 2 sigma^2)
      y:  phi1 = sqrt(k^2 - 2 sigma^2)
    """
    if which == "x":
        phi1 = np.sqrt(kappa**2 + 2*sigma**2)
    elif which == "y":
        phi1 = csqrt(kappa**2 - 2*sigma**2) # may be complex too
    else:
        raise ValueError("which must be 'x' or 'y'")
    phi2 = (kappa + phi1)/2.0
    phi3 = 2*kappa*theta/(sigma**2)
    return phi1, phi2, phi3

def AB_z(tau, which, kappa, theta, sigma):
    """
    A_z, B_z formulas with tau = T - t
    """
    phi1, phi2, phi3 = _phi_triplet(which, kappa, theta, sigma)
    e1 = np.exp(phi1*tau)
    e2 = np.exp(phi2*tau)
    denom = phi2*(e1 - 1.0) + phi1
    A = (phi1*e2/denom)**phi3
    B = (e1 - 1.0)/denom
    return A, B

def bond_price_xy_analytical(x_t, y_t, tau, params_x, params_y):
    """
    P = A_x exp(-B_x x_t) * A_y exp(+B_y y_t)
    """
    kx, thetax, sigx = params_x
    ky, thetay, sigy = params_y
    A_x, B_x = AB_z(tau, "x", kx, thetax, sigx)
    A_y, B_y = AB_z(tau, "y", ky, thetay, sigy)
    P = A_x*np.exp(-B_x*x_t) * A_y*np.exp(+B_y*y_t)
    return P.real if np.iscomplexobj(P) else float(P)

def bond_price_xy_mc(x_t, y_t, tau, params_x, params_y, dt=0.01, n_paths=100000, seed_x=123, seed_y=456):
    # This function can also be used for given x and y with slight adjustments
    kx, thetax, sigx = params_x
    ky, thetay, sigy = params_y
    x_paths = cir_simulate(x_t, kx, thetax, sigx, tau, dt, n_paths, rng=np.random.default_rng(seed_x))
    y_paths = cir_simulate(y_t, ky, thetay, sigy, tau, dt, n_paths, rng=np.random.default_rng(seed_y))
    int_x = cumulative_trapz_on_grid(x_paths, dt)[:,-1]
    int_y = cumulative_trapz_on_grid(y_paths, dt)[:,-1]
    disc = np.exp(-(int_x - int_y))
    price = float(np.mean(disc))
    stderr = float(np.std(disc, ddof=1)/np.sqrt(n_paths))
    return price, stderr

#######
# Plotting functions 
#######

def plot_cir_combined(
    time_grid,
    x_paths, y_paths, z_paths,
    x_lower, x_upper,
    y_lower, y_upper,
    z_lower, z_upper,
    *,
    m=12,
    highlight_idx=0,
    figsize=(12, 7),
    show_zero_line=True,
    legend_ncol=2,
):
    """
    Plot z = x - y, with x and y.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw bands first (so lines are on top)
    ax.fill_between(time_grid, x_lower, x_upper, color="#1f77b4", alpha=0.18, label="x 90% band")
    ax.fill_between(time_grid, y_lower, y_upper, color="#2ca02c", alpha=0.18, label="y 90% band")
    ax.fill_between(time_grid, z_lower, z_upper, color="#7f7f7f", alpha=0.15, label="z 90% band")

    # A few faint paths to show dispersion
    m_eff = min(m, len(x_paths), len(y_paths), len(z_paths))
    for i in range(m_eff):
        ax.plot(time_grid, x_paths[i], color="#1f77b4", linewidth=1.0, alpha=0.15)
        ax.plot(time_grid, y_paths[i], color="#2ca02c", linewidth=1.0, alpha=0.15)
        ax.plot(time_grid, z_paths[i], color="#7f7f7f", linewidth=1.0, alpha=0.12)

    # Highlighted path (same index so z matches x - y)
    idx = int(highlight_idx)
    ax.plot(time_grid, x_paths[idx], color="#1f77b4", linewidth=2.6, label="x highlighted")
    ax.plot(time_grid, y_paths[idx], color="#2ca02c", linestyle="--", linewidth=2.6, label="y highlighted")
    ax.plot(time_grid, z_paths[idx], color="#000000", linestyle="-.", linewidth=2.6, label="r = x − y highlighted")

    if show_zero_line:
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)

    ax.set_title("CIR simulations: r(t) = x(t) - y(t)")
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Interest Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=legend_ncol)
    ax.margins(x=0)
    plt.tight_layout()

    # return fig, ax

def plot_cir_x_only(
    time_grid,
    x_paths,
    x_lower, x_upper,
    x_mean,
    theta_x,
    *,
    fan_n=24,
    highlight_idx=0,
    figsize=(12, 6),
):
    """
    Plot x process only
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(time_grid, x_lower, x_upper, color="#b43d1f", alpha=0.20, label="90% band (exact)")

    # small fan
    fan_eff = min(fan_n, len(x_paths))
    for i in range(fan_eff):
        ax.plot(time_grid, x_paths[i], color="#1f77b4", linewidth=0.9, alpha=0.12)

    idx = int(highlight_idx)
    ax.plot(time_grid, x_paths[idx], color="#1f77b4", linewidth=2.8, label="highlighted path")
    ax.plot(time_grid, x_mean, color="red", linewidth=1.8, label="theoretical mean")
    ax.axhline(theta_x, color="red", linestyle=":", linewidth=1.2, alpha=0.8, label="mean")

    ax.set_title("CIR simulations: x(t)")
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Interest Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.margins(x=0)
    plt.tight_layout()


def plot_cir_bond_price(
    maturities,
    p_analytical,
    mc_prices,
    mc_std_err,
    *,
    figsize=(10, 6),
):
    """
    Plot analytical CIR zero-coupon bond prices vs Monte Carlo estimates
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(maturities, p_analytical, marker="o", linewidth=2.0, label="Analytical (CIR)")
    ax.errorbar(
        maturities, mc_prices, yerr=mc_std_err,
        fmt="s", capsize=3, label="Monte Carlo (±1 SE)"
    )
    ax.set_title("Zero-coupon bond price under CIR: analytical vs Monte Carlo")
    ax.set_xlabel("Maturity T in years")
    ax.set_ylabel("Price P(0,T)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
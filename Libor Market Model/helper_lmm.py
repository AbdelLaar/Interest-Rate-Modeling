import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution, curve_fit, minimize_scalar
from scipy.integrate import quad
from scipy.interpolate import griddata


def phi_abcd_volatility(T_j, t, phi, a, b, c, d):
    time_diff = T_j - t
    volatility = phi * ((a * time_diff + d) * np.exp(-b * time_diff) + c)
    return volatility

def phi_abcd_model(params, T_array, t=0):
    n_periods = len(T_array)
    phi_values = params[:n_periods]
    a, b, c, d = params[n_periods:]
    
    volatilities = np.zeros(n_periods)
    for i, T_j in enumerate(T_array):
        volatilities[i] = phi_abcd_volatility(T_j, t, phi_values[i], a, b, c, d)
    
    return volatilities

def objective_function(params, T_array, market_vols):
    model_vols = phi_abcd_model(params, T_array)
    return np.sum((model_vols - market_vols)**2)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calibration Method: first estimate ABCD, then phi
def calibrate_method3(T_array, market_vols):
    n_periods = len(T_array)
    
    def objective_abcd(abcd_params):
        phi_values = np.ones(n_periods)
        params = np.concatenate([phi_values, abcd_params])
        return objective_function(params, T_array, market_vols)
    
    abcd_bounds = [(0.001, 2.0), (0.001, 5.0), (0.0, 1.0), (0.0, 1.0)]
    abcd_result = differential_evolution(objective_abcd, abcd_bounds,
                                       maxiter=200, seed=42)
    a_opt, b_opt, c_opt, d_opt = abcd_result.x
    
    def abcd_function(T_j, phi):
        return phi_abcd_volatility(T_j, 0, phi, a_opt, b_opt, c_opt, d_opt)
    
    phi_values = np.zeros(n_periods)
    for i, T_j in enumerate(T_array):
        def single_vol_func(T, phi):
            return abcd_function(T, phi)
        
        popt, _ = curve_fit(single_vol_func, [T_j], [market_vols[i]],
                           bounds=(0.1, 5.0), maxfev=1000)
        phi_values[i] = popt[0]
    
    final_params = np.concatenate([phi_values, [a_opt, b_opt, c_opt, d_opt]])
    
    class Result:
        def __init__(self, x, fun):
            self.x = x
            self.fun = fun
            self.success = True
    
    final_objective = objective_function(final_params, T_array, market_vols)
    return Result(final_params, final_objective)

def correlation_matrix_parametrization(beta: float, n_rates: int) -> np.ndarray:
    correlation_matrix = np.zeros((n_rates, n_rates))
    for i in range(n_rates):
        for j in range(n_rates):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                correlation_matrix[i, j] = np.exp(-beta * abs(i - j))
    
    return correlation_matrix


### Plotting functions

def plot_phi_abcd_volatility_function2(
    calibration_results,
    market_vols_clean,
    T_original,
    best_method,
    T_range=None
):
    if T_range is None:
        T_range = np.linspace(0.1, 6.0, 200)

    best_result = calibration_results[best_method]
    a, b, c, d = best_result['a'], best_result['b'], best_result['c'], best_result['d']
    phi_values = best_result['phi']

    phi_avg = float(np.mean(phi_values))
    model_vols_extended = np.array([
        phi_abcd_volatility(T_val, 0.0, phi_avg, a, b, c, d) for T_val in T_range
    ], dtype=float)

    individual_model_vols = np.array([
        phi_abcd_volatility(T_val, 0.0, phi_i, a, b, c, d)
        for T_val, phi_i in zip(T_original, phi_values)
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T_range, model_vols_extended, linewidth=3,
            label=f'Calibrated Phi-ABCD (avg Phi={phi_avg:.3f})')
    ax.plot(T_original, market_vols_clean, 'o', markersize=6,
            label='Market Caplet Volatilities')
    ax.plot(T_original, individual_model_vols, '^', markersize=6,
            label='Individual Calibrated Points')

    ax.set_xlabel('Time T in years')
    ax.set_ylabel('Volatility')
    ax.set_title('Calibrated Phi-ABCD vs Market Data')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return T_range, model_vols_extended, individual_model_vols


def plot_correlation_matrix(corr_matrix, title="Correlation Matrix"):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title(title)
    plt.xlabel('Forward Rate Index')
    plt.ylabel('Forward Rate Index')
    plt.show()

def plot_multiple_rates(time_grid, paths, rate_maturities, rate_index,
                     selected_path=0, max_rates=8, tau = 0.25
                     ):
    fig, ax1 = plt.subplots(1, figsize=(16, 8))
    
    n_rates_to_plot = min(max_rates, paths.shape[2])
    colors = plt.cm.tab10(np.linspace(0, 1, n_rates_to_plot))
    
    for rate_idx in rate_index:
        # Extract path for this rate, handling None values
        rate_path = []
        time_points = []
        
        for t_idx in range(len(time_grid)):
            rate_value = paths[selected_path, t_idx, rate_idx]
            if rate_value is not None:
                rate_path.append(rate_value)
                time_points.append(time_grid[t_idx])
            else:
                # rate has matured, so stop plotting
                break
        
        if len(rate_path) > 0:
            ax1.plot(time_points, rate_path, color=colors[rate_idx], linewidth=2.5, 
                   label=f'Rate {rate_idx+1} (T={rate_maturities[rate_idx]:.2f}y)',
                   marker='o', markersize=3, markevery=max(1, len(time_points)//20))
            
            maturity_time = rate_maturities[rate_idx] # -tau
            if len(time_points) > 0:
                last_rate = rate_path[-1]
                
                ax1.plot(maturity_time, last_rate, 'x', color=colors[rate_idx], 
                       markersize=10, markeredgewidth=3)
                ax1.axvline(x=maturity_time, color=colors[rate_idx], linestyle=':', 
                          alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Time in years', fontsize=12)
    ax1.set_ylabel('Forward Rate', fontsize=12)
    ax1.set_title(f'Forward Rates Simulation', 
                fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Multiple simulations of one rate
def plot_simulated_rate2(
    time_grid,
    paths,
    rate_maturities,
    selected_path=0,
    selected_rate=0,
    n_paths_to_show=100,
    percentile_5=None,
    percentile_95=None,
    # tau=0.25,
):
    if percentile_5 is None or percentile_95 is None:
        percentile_5 = np.nanpercentile(paths, 5, axis=0)
        percentile_95 = np.nanpercentile(paths, 95, axis=0)

    fig, ax1 = plt.subplots(1, figsize=(16, 8))

    for path_idx in range(min(n_paths_to_show, paths.shape[0])):
        rate_series = paths[path_idx, :, selected_rate]
        mask = np.isfinite(rate_series)
        if not np.any(mask):
            continue
        if path_idx == selected_path:
            ax1.plot(time_grid[mask], rate_series[mask], color='red', alpha=0.9, linewidth=3,
                     label=f'Highlighted Path {selected_path}', zorder=5)
        else:
            ax1.plot(time_grid[mask], rate_series[mask], color='blue', alpha=0.4, linewidth=0.7)

    maturity_time = rate_maturities[selected_rate] 
    ax1.axvline(x=maturity_time, color='black', linestyle='--',
                linewidth=2, label=f'Fixing at {maturity_time:.2f}y', zorder=2)

    p5 = percentile_5[:, selected_rate]
    p95 = percentile_95[:, selected_rate]
    band_mask = (time_grid <= maturity_time) & np.isfinite(p5) & np.isfinite(p95)
    if np.any(band_mask):
        ax1.fill_between(time_grid[band_mask], p5[band_mask], p95[band_mask],
                         alpha=0.3, color='red', label='90% Confidence Band')

    ax1.set_xlabel('Time in years', fontsize=12)
    ax1.set_ylabel('Forward Rate ', fontsize=12)
    ax1.set_title(f'Forward rates simulation',
                  fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ZCB Part

def plot_zcb_price_evolution(zcb_pricer, time_grid, paths, bond_maturity, n_paths_to_show=100):
    fig, ax1 = plt.subplots( figsize=(12, 8))
    
    pricing_times = time_grid[::10] # Sample every 10th time point
    
    for path_idx in range(min(n_paths_to_show, paths.shape[0])):
        zcb_prices_path = []
        valid_times = []
        
        for pricing_time in pricing_times:
            if pricing_time >= bond_maturity:
                break 
                
            try:
                pricing_idx = np.argmin(np.abs(time_grid - pricing_time))
                current_forward_rates = np.zeros(len(zcb_pricer.rate_maturities))
                
                for rate_idx in range(len(zcb_pricer.rate_maturities)):
                    rate_value = paths[path_idx, pricing_idx, rate_idx]
                    if rate_value is not None:
                        current_forward_rates[rate_idx] = rate_value
                    else:
                        current_forward_rates[rate_idx] = 0.0
                
                zcb_price = zcb_pricer.analytical_zcb_price(
                    pricing_time, bond_maturity, current_forward_rates
                )
                zcb_prices_path.append(zcb_price)
                valid_times.append(pricing_time)
                
            except:
                continue
        
        if len(zcb_prices_path) > 0:
            if path_idx == 0:
                ax1.plot(valid_times, zcb_prices_path, 'r-', linewidth=2, 
                        label='First Path', alpha=0.8)
            else:
                ax1.plot(valid_times, zcb_prices_path, 'b-', linewidth=0.5, alpha=0.3)
    
    ax1.set_xlabel('Time in years')
    ax1.set_ylabel('Zero Coupon Bond Price')
    ax1.set_title(f'Zero Coupon Bond Price Evolution (Maturity = {bond_maturity:.1f}y)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots( figsize=(12, 8))
    sample_times = [0.5, 1.0, 1.5, 2.0]
    sample_times = [t for t in sample_times if t < bond_maturity]
    
    for i, pricing_time in enumerate(sample_times):
        mc_result = zcb_pricer.mc_zcb_price_from_paths(
            time_grid, paths, pricing_time, bond_maturity
        )
        
        ax2.hist(mc_result['all_prices'], bins=50, alpha=0.6, 
                label=f't = {pricing_time:.1f}y', density=True)
    
    ax2.set_xlabel('Zero Coupon Bond Price')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Zero Coupon Bond Price Distribution (Maturity = {bond_maturity:.1f}y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_zcb_comparison_table(comparison_results):
    print(f"{'Pricing Time':<12} {'Bond Maturity':<12} {'Analytical':<12} {'MC Mean':<12} {'Difference':<12}")
    print("-"*80)
    
    for pricing_time, bonds in comparison_results.items():
        for bond_maturity, result in bonds.items():
            print(f"{pricing_time:<12.2f} {bond_maturity:<12.2f} "
                  f"{result['analytical']:<12.6f} {result['mc_mean']:<12.6f} "
                  f"{result['difference']:<12.12f}")

# Cap Pricing

def calculate_integrated_variance(T_j, current_time, phi_j, a, b, c, d):
    if T_j <= current_time:
        return 0.0
    
    def volatility_squared(s):
        vol = phi_abcd_volatility(T_j, s, phi_j, a, b, c, d)
        return vol**2
    
    try:
        integral_result, _ = quad(volatility_squared, current_time, T_j, limit=100)
        time_to_maturity = T_j - current_time
        integrated_variance = integral_result / time_to_maturity   
        return integrated_variance
        
    except Exception as e:
        print(f"Integration failed for T_j={T_j}, t={current_time}: {e}")
        mid_time = (current_time + T_j) / 2
        avg_vol = phi_abcd_volatility(T_j, mid_time, phi_j, a, b, c, d)
        return avg_vol**2

def black_caplet_price(F_j, K, T_j, phi_j, abcd_params, current_time=0, tau=0.25):
    if T_j <= current_time:
        return 0.0
    
    a, b, c, d = abcd_params
    
    integrated_variance = calculate_integrated_variance(T_j, current_time, phi_j, a, b, c, d)
    
    if integrated_variance <= 0:
        return 0.0
    
    time_to_maturity = T_j - current_time
    
    total_variance = integrated_variance * time_to_maturity
    
    effective_volatility = np.sqrt(integrated_variance)
    
    d1 = (np.log(F_j / K) + 0.5 * total_variance) / (effective_volatility * np.sqrt(time_to_maturity))
    d2 = d1 - effective_volatility * np.sqrt(time_to_maturity)
    
    caplet_value = tau * (F_j * norm.cdf(d1) - K * norm.cdf(d2))
    
    return caplet_value

def price_cap_monte_carlo_spot_measure(
    time_grid,
    paths,
    strike, 
    maturities,
    tau=0.25,
    notional=1.0,
    return_se=False,
):
    paths = np.asarray(paths)
    time_grid = np.asarray(time_grid)
    maturities = np.asarray(maturities)
    n_paths, _, n_rates = paths.shape

    if np.isscalar(strike):
        K = np.full(n_rates, float(strike))
    else:
        K = np.asarray(strike, dtype=float)
        if K.shape[0] != n_rates:
            raise ValueError("strike must be scalar or have length n_rates")

    fix_times = maturities - tau
    fix_idx = np.array([int(np.argmin(np.abs(time_grid - t))) for t in fix_times])
    pay_idx = np.array([int(np.argmin(np.abs(time_grid - t))) for t in maturities])

    # % if np.any(pay_idx < fix_idx):
    # %     raise ValueError("Payment index occurs before fixing index for some caplets.")

    L_fix = paths[np.arange(n_paths)[:, None], fix_idx[None, :], np.arange(n_rates)[None, :]]

    payoffs = notional * tau * np.maximum(L_fix - K[None, :], 0.0) 

    caplet_prices = np.zeros(n_rates)
    if return_se:
        caplet_ses = np.zeros(n_rates)
        
    for i in range(n_rates):
        beta_minus_1 = i
        if beta_minus_1 > 0:
            money_market = np.ones(n_paths)
            for j in range(beta_minus_1):
                L_j_at_Tj = paths[:, fix_idx[j], j]
                L_j_at_Tj = np.where(L_j_at_Tj == L_j_at_Tj, L_j_at_Tj, 0.0) 
                money_market *= (1.0 + tau * L_j_at_Tj)
        else:
            money_market = np.ones(n_paths)

        L_i_at_Ti = L_fix[:, i]
        L_i_at_Ti = np.where(L_i_at_Ti == L_i_at_Ti, L_i_at_Ti, 0.0)  # Handle NaN
        full_money_market = money_market * (1.0 + tau * L_i_at_Ti)
        
        discounted_payoffs = payoffs[:, i] / full_money_market
        caplet_prices[i] = np.mean(discounted_payoffs)
        
        if return_se:
            caplet_ses[i] = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)

    cap_price = float(np.sum(caplet_prices))

    if return_se:
        return cap_price, caplet_prices, caplet_ses

    return cap_price, caplet_prices

def discount_factors_from_forwards(forwards, tau=0.25):
    one_plus = 1.0 + tau * np.asarray(forwards, float)
    P = 1.0 / np.cumprod(one_plus)
    return P

def price_cap_black_with_P0T(forward_rates, phi_values, abcd_params, strike, maturities, P0T, tau=0.25):
    caplets, vars_ = [], []
    for i, (F_j, T_j, phi_j) in enumerate(zip(forward_rates, maturities, phi_values)):
        vbar = calculate_integrated_variance(T_j, 0.0, phi_j, *abcd_params); vars_.append(vbar)
        undisc = black_caplet_price(F_j, strike, T_j, phi_j, abcd_params, tau=tau)
        caplets.append(undisc * P0T[i])
    caplets = np.array(caplets, float)
    return float(caplets.sum()), caplets, np.array(vars_)

def plot_caplet_prices_vs_maturity(
    maturities,
    caplet_prices_mc,
    caplet_prices_black=None,
    caplet_se=None, 
    title="Caplet Price Evolution",
):
    maturities = np.asarray(maturities, float)
    mc = np.asarray(caplet_prices_mc, float)

    fig = plt.figure(figsize=(9, 5.5))
    if caplet_se is not None:
        caplet_se = np.asarray(caplet_se, float)
        plt.errorbar(maturities, mc, yerr=caplet_se, fmt='o', capsize=3, label='MC (+-1 SE)')
    else:
        plt.plot(maturities, mc, 'o', label='MC')

    if caplet_prices_black is not None:
        blk = np.asarray(caplet_prices_black, float)
        plt.plot(maturities, blk, '-', linewidth=2.0, label="Black")

    plt.xlabel("Maturity in years")
    plt.ylabel("Price in bp")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def black_caplet_implied_vol(caplet_price, forward_rate, strike, maturity, discount_factor, tau=0.25, initial_guess=0.2):    
    def black_caplet_price_formula(vol):
        if vol <= 0:
            return 0.0
        
        sqrt_T = np.sqrt(maturity)
        d1 = (np.log(forward_rate / strike) + 0.5 * vol**2 * maturity) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T
        
        price = tau * discount_factor * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
        return price
    
    def objective(vol):
        model_price = black_caplet_price_formula(vol)
        return (model_price - caplet_price)**2
    
    if caplet_price <= 0:
        return 0.0
    
    if forward_rate <= strike:
        if caplet_price < 1e-10:
            return 0.0
    
    try:
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        
        if result.success and result.fun < 1e-8:
            return result.x
        else:
            result2 = minimize_scalar(objective, bounds=(0.0001, 10.0), method='bounded')
            if result2.success:
                return result2.x
            else:
                return np.nan
    except:
        return np.nan

def calculate_caplet_implied_vols(caplet_prices, forward_rates, strike, maturities, discount_factors, tau=0.25):
    n_caplets = len(caplet_prices)
    implied_vols = np.zeros(n_caplets)
    
    for i in range(n_caplets):
        implied_vols[i] = black_caplet_implied_vol(
            caplet_prices[i],
            forward_rates[i], 
            strike,
            maturities[i] - tau,  # fixing time T_i = T_{i+1} - tau
            discount_factors[i],
            tau
        )    
    return implied_vols

def plot_caplet_implied_volatilities(
    maturities,
    mc_caplet_prices,
    black_caplet_prices,
    forward_rates,
    strike_rate,
    discount_factors,
    market_caplet_vols=None,
    tau=0.25
):
    fixing_times = maturities - tau
    
    mc_implied_vols = calculate_caplet_implied_vols(
        mc_caplet_prices, forward_rates, strike_rate, maturities, discount_factors, tau
    )
    
    black_implied_vols = calculate_caplet_implied_vols(
        black_caplet_prices, forward_rates, strike_rate, maturities, discount_factors, tau
    )
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(fixing_times, mc_implied_vols, 'o-', linewidth=2, markersize=6, 
             label='MC Implied Volatilities', color='blue')
    
    if market_caplet_vols is not None:
        # Assume market vols start from the second caplet
        market_fixing_times = fixing_times[1:len(market_caplet_vols)+1] if len(market_caplet_vols) < len(fixing_times) else fixing_times
        plt.plot(market_fixing_times, market_caplet_vols, '^-', linewidth=2, markersize=6, 
                 label='Market Caplet Volatilities', color='green')
    
    plt.xlabel('Time in years', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.title(f'Caplet Implied Volatilities\n(Strike = {strike_rate:.1%})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    mc_valid = mc_implied_vols[~np.isnan(mc_implied_vols)]
    black_valid = black_implied_vols[~np.isnan(black_implied_vols)]
    
    if len(mc_valid) > 0 and len(black_valid) > 0:
        rmse = np.sqrt(np.mean((mc_valid - black_valid[:len(mc_valid)])**2))
        max_diff = np.max(np.abs(mc_valid - black_valid[:len(mc_valid)]))
        
        textstr = f'RMSE: {rmse:.4f}\nMax Diff: {max_diff:.4f}'
        print(textstr)
    
    plt.tight_layout()
    plt.show()

    return mc_implied_vols, black_implied_vols

# Swaptions

def calculate_swap_rate_and_weights(forward_rates, alpha_idx, beta_idx, tau= 0.25):
    relevant_forwards = forward_rates[alpha_idx+1:beta_idx+1]
    n_forwards = len(relevant_forwards)
    
    if n_forwards == 0:
        return 0.0, np.array([])
    
    discount_factors = np.zeros(n_forwards)
    for i in range(n_forwards):
        discount_factors[i] = np.prod(1 / (1 + tau * relevant_forwards[:i+1]))
    
    annuity = tau * np.sum(discount_factors)
    
    if annuity > 0:
        swap_rate = (1 - discount_factors[-1]) / annuity
    else:
        swap_rate = 0.0
    
    weights = (tau * discount_factors) / annuity if annuity > 0 else np.zeros(n_forwards)
    
    return swap_rate, weights

def rebonato_swaption_volatility(forward_rates, phi_values, abcd_params, correlation_matrix, 
                                          alpha_idx, beta_idx, T_alpha, tau=0.25):
    swap_rate, weights = calculate_swap_rate_and_weights(forward_rates, alpha_idx, beta_idx, tau)
    
    if len(weights) == 0 or swap_rate == 0:
        return 0.0
    
    relevant_forwards = forward_rates[alpha_idx+1:beta_idx+1]
    relevant_phi = phi_values[alpha_idx+1:beta_idx+1]
    n_rates = len(relevant_forwards)
    
    a, b, c, d = abcd_params
    variance_sum = 0.0
    
    for i in range(n_rates):
        for j in range(n_rates):
            global_i = alpha_idx + 1 + i
            global_j = alpha_idx + 1 + j
            
            rho_ij = correlation_matrix[global_i, global_j]
            
            T_i = (global_i + 1) * tau
            T_j = (global_j + 1) * tau
            phi_i = relevant_phi[i]
            phi_j = relevant_phi[j]
            
            def integrand(t):
                sigma_i_t = phi_abcd_volatility(T_i, t, phi_i, a, b, c, d)
                sigma_j_t = phi_abcd_volatility(T_j, t, phi_j, a, b, c, d)
                
                return sigma_i_t * sigma_j_t * rho_ij
            
            try:
                integral_result, _ = quad(integrand, 0, T_alpha, limit=50)
            except:
                sigma_i_avg = phi_abcd_volatility(T_i, T_alpha/2, phi_i, a, b, c, d)
                sigma_j_avg = phi_abcd_volatility(T_j, T_alpha/2, phi_j, a, b, c, d)
                integral_result = sigma_i_avg * sigma_j_avg * rho_ij * T_alpha
            
            contribution = (weights[i] * weights[j] * relevant_forwards[i] * relevant_forwards[j] * 
                          integral_result)
            
            variance_sum += contribution
    
    if swap_rate > 0:
        swaption_variance = variance_sum / (swap_rate ** 2)
        swaption_volatility = np.sqrt(max(0, swaption_variance))
    else:
        swaption_volatility = 0.0
    
    return swaption_volatility

def compute_rebonato_black_vol_surface(
    forward_rates,
    phi_values, 
    abcd_params,
    correlation_matrix,
    maturities, 
    option_expiries, 
    swap_tenors,
    tau=0.25
):
    option_expiries = np.asarray(option_expiries, float)
    swap_tenors     = np.asarray(swap_tenors, float)
    maturities      = np.asarray(maturities, float)
    T_fix = maturities - tau

    V = np.full((len(option_expiries), len(swap_tenors)), np.nan, dtype=float)

    for i, Texp in enumerate(option_expiries):
        ix = np.where(np.isclose(T_fix, Texp))[0]
        if ix.size == 0:
            alpha_idx = int(np.argmin(np.abs(T_fix - Texp)))
            Talpha = T_fix[alpha_idx]
        else:
            alpha_idx = int(ix[0])
            Talpha = Texp

        for j, tenor in enumerate(swap_tenors):
            Tbeta = Talpha + tenor
            iy = np.where(np.isclose(maturities, Tbeta))[0]
            if iy.size == 0:
                continue
            beta_idx = int(iy[0])
            if beta_idx <= alpha_idx:  # needs at least one payment
                continue

            vol_path = rebonato_swaption_volatility(
                forward_rates, phi_values, abcd_params, correlation_matrix,
                alpha_idx, beta_idx, Talpha, tau=tau
            )
            V[i, j] = vol_path / np.sqrt(max(Talpha, 1e-12))

    return V

def calculate_model_swaption_matrix(
    forward_rates,
    phi_values, 
    abcd_params, 
    correlation_matrix,
    option_expiries, 
    swap_tenors, 
    maturities,
    tau=0.25,
    return_black_vol=True 
):
    forward_rates  = np.asarray(forward_rates,  float)
    phi_values     = np.asarray(phi_values,     float)
    maturities     = np.asarray(maturities,     float)
    option_expiries= np.asarray(option_expiries,float)
    swap_tenors    = np.asarray(swap_tenors,    float)

    n_forwards = forward_rates.shape[0]
    T_fix = maturities - tau 
    V = np.full((len(option_expiries), len(swap_tenors)), np.nan, dtype=float)

    valid = 0
    for i, Talpha in enumerate(option_expiries):
        ix = np.where(np.isclose(T_fix, Talpha))[0]
        if ix.size == 0:
            alpha_idx = int(np.argmin(np.abs(T_fix - Talpha)))
            Talpha_eff = T_fix[alpha_idx]
        else:
            alpha_idx = int(ix[0])
            Talpha_eff = Talpha

        for j, tenor in enumerate(swap_tenors):
            Tbeta = Talpha_eff + tenor
            iy = np.where(np.isclose(maturities, Tbeta))[0]
            if iy.size == 0:
                continue
            beta_idx = int(iy[0])

            if beta_idx <= alpha_idx:
                continue

            vol_path = rebonato_swaption_volatility(
                forward_rates, phi_values, abcd_params, correlation_matrix,
                alpha_idx, beta_idx, Talpha_eff, tau=tau
            )

            if return_black_vol:
                if Talpha_eff <= 0:
                    continue
                V[i, j] = vol_path / np.sqrt(Talpha_eff) 
            else:
                V[i, j] = vol_path

            valid += 1

    total = V.size
    print(f"Calculated {valid}/{total} swaptions "
          f"({100.0*valid/total:.1f}%). NaNs: {np.isnan(V).sum()}")
    return V

def plot_swaption_matrix(model_vols, option_expiries, swap_tenors):
    fig, ax1 = plt.subplots(figsize=(8, 8))

    masked_vols = np.ma.masked_where(np.isnan(model_vols), model_vols)
    
    im1 = ax1.imshow(masked_vols, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title(f'Model Swaption Volatilities')
    ax1.set_xlabel('Swap Tenor in years')
    ax1.set_ylabel('Option Expiry in years')
    
    x_ticks = range(0, len(swap_tenors), 4)
    y_ticks = range(0, len(option_expiries), 4)
    
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{swap_tenors[i]:.1f}' for i in x_ticks])
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f'{option_expiries[i]:.1f}' for i in y_ticks])
    
    plt.colorbar(im1, ax=ax1, label='Swaption Volatility')

    plt.tight_layout()
    plt.show()

    # fig, ax2 = plt.subplots(figsize=(7, 8))

    # valid_mask = ~np.isnan(model_vols)
    # binary_image = valid_mask.astype(int)
    
    # im2 = ax2.imshow(binary_image, cmap='RdYlGn', aspect='auto', origin='lower')
    # ax2.set_title('Swaption Map\n(Green = Valid, Red = Invalid)')
    # ax2.set_xlabel('Swap Tenor in years')
    # ax2.set_ylabel('Option Expiry in years')
    
    # ax2.set_xticks(x_ticks)
    # ax2.set_xticklabels([f'{swap_tenors[i]:.1f}' for i in x_ticks])
    # ax2.set_yticks(y_ticks)
    # ax2.set_yticklabels([f'{option_expiries[i]:.1f}' for i in y_ticks])
    
    # plt.tight_layout()
    # plt.show()

def plot_3d_volatility_surface(model_swaption_vols, option_expiries, swap_tenors):
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    valid_mask = ~np.isnan(model_swaption_vols)
    
    if np.sum(valid_mask) > 3:
        valid_indices = np.where(valid_mask)
        y_valid = option_expiries[valid_indices[0]]
        x_valid = swap_tenors[valid_indices[1]]
        z_valid = model_swaption_vols[valid_indices]
        
        scatter = ax.scatter(x_valid, y_valid, z_valid, c=z_valid, cmap='viridis', 
                           s=80, alpha=0.9, edgecolors='black', linewidth=0.5,
                           label=f'Model Volatilities ({len(y_valid)} points)')
        
        if len(z_valid) > 6:
            try:
                xi = np.linspace(option_expiries.max(), option_expiries.min(), 30)
                yi = np.linspace(swap_tenors.max(), swap_tenors.min(), 25)
                Xi, Yi = np.meshgrid(xi, yi, indexing='ij')
                
                Zi = griddata((x_valid, y_valid), z_valid, (Xi, Yi), 
                             method='cubic', fill_value=np.nan)
                
                surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.7, 
                                     linewidth=0, antialiased=True, shade=True,
                                     rcount=40, ccount=40)
                                
            except Exception as e:                
                try:
                    X, Y = np.meshgrid(swap_tenors, option_expiries, indexing='ij')
                    Z = model_swaption_vols.copy()
                    
                    surf = ax.plot_wireframe(X, Y, Z, alpha=0.6, color='blue', linewidth=1.5)
                    
                except Exception as e2:
                    surf = None
        
        if 'surf' in locals() and surf is not None:
            try:
                cbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
                cbar.set_label('Swaption Volatility', fontsize=12)
            except:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
                cbar.set_label('Swaption Volatility', fontsize=12)
        else:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label('Swaption Volatility', fontsize=12)
        
    else:
        print("Not enough valid volatility points for surface creation")
        return
    
    ax.set_xlabel('Option Expiry in years', fontsize=12, labelpad=10)
    ax.set_ylabel('Swap Tenor in years', fontsize=12, labelpad=10)
    ax.set_zlabel('Swaption Volatility', fontsize=12, labelpad=10)
    
    ax.set_title('Swaption Volatility Surface', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=25, azim=45)
    
    ax.grid(True, alpha=0.3)
    
    if len(z_valid) > 0:
        z_margin = (np.max(z_valid) - np.min(z_valid)) * 0.1
        ax.set_zlim(np.min(z_valid) - z_margin, np.max(z_valid) + z_margin)

    plt.tight_layout()
    plt.show()

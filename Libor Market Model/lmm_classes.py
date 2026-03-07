import numpy as np
from scipy.linalg import cholesky
import helper_lmm as hl

class LMMSimulator:
    def __init__(self, initial_rates, phi_values, abcd_params, correlation_matrix, rate_maturities):
        self.initial_rates = np.array(initial_rates)
        self.phi_values = np.array(phi_values)
        self.a, self.b, self.c, self.d = abcd_params
        self.correlation_matrix = correlation_matrix
        self.rate_maturities = np.array(rate_maturities)
        self.n_rates = len(initial_rates)
        self.tau = 0.25
        
        try:
            self.chol_matrix = cholesky(correlation_matrix, lower=True)
        except np.linalg.LinAlgError:
            print("Warning: Correlation matrix not positive definite. Using eigenvalue regularization.")
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)
            reg_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            self.chol_matrix = cholesky(reg_corr, lower=True)

    def get_volatility(self, rate_idx, current_time):
        T_pay = self.rate_maturities[rate_idx]    
        T_fix = T_pay - self.tau 
        if current_time >= T_fix: 
            return 0.0
        phi_j = self.phi_values[rate_idx]
        return hl.phi_abcd_volatility(T_fix, current_time, phi_j, self.a, self.b, self.c, self.d)

    def simulate_all_rates_jointly(
        self,
        n_paths=1000,
        T_max=None,
        n_time_steps=200,
        seed=None,
        freeze_mode="constant",
        return_fixings=True
    ):
        if seed is not None:
            np.random.seed(seed)

        if T_max is None:
            T_max = float(self.rate_maturities[-1])

        time_grid = np.linspace(0.0, T_max, n_time_steps + 1)
        dt = T_max / n_time_steps
        sqrt_dt = np.sqrt(dt)

        P = int(n_paths)
        R = int(self.n_rates)

        paths = np.zeros((P, n_time_steps + 1, R), dtype=float)
        paths[:, 0, :] = self.initial_rates[None, :]

        T_pay = np.asarray(self.rate_maturities, dtype=float) 
        T_fix = T_pay - self.tau
        L_fix_snap = np.full((P, R), np.nan, dtype=float) if return_fixings else None

        L_chol = self.chol_matrix

        for p in range(P):
            for t_idx in range(n_time_steps):
                t = time_grid[t_idx]

                # correlated shocks 
                z = np.random.normal(0.0, 1.0, R)
                dW = L_chol @ z  # shape (R,)

                cur = paths[p, t_idx, :].copy()

                for j in range(R):
                    if t >= T_pay[j]:
                        paths[p, t_idx + 1, j] = paths[p, t_idx, j]
                        continue

                    if t >= T_fix[j]:
                        if return_fixings and np.isnan(L_fix_snap[p, j]):
                            L_fix_snap[p, j] = paths[p, t_idx, j]
                        if freeze_mode == "constant":
                            paths[p, t_idx + 1, j] = paths[p, t_idx, j]
                        elif freeze_mode == "nan":
                            paths[p, t_idx + 1, j] = np.nan
                        else:
                            raise ValueError("freeze_mode must be 'constant' or 'nan'")
                        continue

                    sigma_j = self.get_volatility(j, t)
                    drift_j = self.calculate_drift(cur, j, t)
                    L_j = max(cur[j], 1e-12)

                    dL = L_j * drift_j * dt + L_j * sigma_j * dW[j] * sqrt_dt
                    new_L = max(L_j + dL, 1e-6)  # small floor to keep 1+tau*L > 0
                    paths[p, t_idx + 1, j] = new_L

            if return_fixings:
                for j in range(R):
                    if np.isnan(L_fix_snap[p, j]):
                        idx = np.searchsorted(time_grid, T_fix[j], side="right") - 1
                        idx = np.clip(idx, 0, n_time_steps)
                        L_fix_snap[p, j] = paths[p, idx, j]

        if return_fixings:
            return time_grid, paths, L_fix_snap
        return time_grid, paths
    
    def calculate_drift(self, current_rates, rate_idx, current_time):
        T_pay = self.rate_maturities[rate_idx]
        T_fix = T_pay - self.tau
        if current_time >= T_fix:
            return 0.0

        q_t = self.get_q_index(current_time)
        vol_j = self.get_volatility(rate_idx, current_time)
        if vol_j <= 0:
            return 0.0

        drift_sum = 0.0
        for k in range(q_t, rate_idx + 1):
            if current_time < self.rate_maturities[k]:
                Lk = current_rates[k]
                if Lk > 0.0:
                    vol_k = self.get_volatility(k, current_time)
                    if vol_k > 0.0:
                        rho = self.correlation_matrix[k, rate_idx]
                        drift_sum += rho * (self.tau * vol_k * Lk) / (1.0 + self.tau * Lk)
        return vol_j * drift_sum


    
    def get_q_index(self, current_time):
        active_indices = np.where(self.rate_maturities >= current_time)[0]
        return active_indices[0] if len(active_indices) > 0 else self.n_rates
    

class ZeroCouponBondPricer:
    def __init__(self, lmm_simulator, forward_rates, rate_maturities, tau=0.25):
        self.lmm_simulator = lmm_simulator
        self.forward_rates = np.array(forward_rates)
        self.rate_maturities = np.array(rate_maturities)
        self.tau = tau
        
    def analytical_zcb_price(self, current_time, bond_maturity, current_forward_rates=None):
        if current_forward_rates is None:
            current_forward_rates = self.forward_rates.copy()
        
        applicable_mask = (self.rate_maturities > current_time) & (self.rate_maturities <= bond_maturity)
        applicable_rates = current_forward_rates[applicable_mask]
        
        if len(applicable_rates) == 0:
            return 1.0  # No discounting needed
        
        discount_factors = 1.0 / (1.0 + self.tau * applicable_rates)
        zcb_price = np.prod(discount_factors)
        
        return zcb_price
    
    def mc_zcb_price_from_paths(self, time_grid, paths, pricing_time, bond_maturity):
        pricing_idx = np.argmin(np.abs(time_grid - pricing_time))
        actual_pricing_time = time_grid[pricing_idx]
        
        n_paths = paths.shape[0]
        zcb_prices = np.zeros(n_paths)
                
        for path_idx in range(n_paths):
            # Extract forward rates at pricing time for this path
            current_forward_rates = np.zeros(len(self.rate_maturities))
            
            for rate_idx in range(len(self.rate_maturities)):
                rate_value = paths[path_idx, pricing_idx, rate_idx]
                if rate_value is not None:
                    current_forward_rates[rate_idx] = rate_value
                else:
                    current_forward_rates[rate_idx] = 0.0  # Rate has matured
            
            zcb_prices[path_idx] = self.analytical_zcb_price(
                actual_pricing_time, bond_maturity, current_forward_rates
            )
        
        mean_price = np.mean(zcb_prices)
        std_price = np.std(zcb_prices)
        ci_lower = np.percentile(zcb_prices, 2.5)
        ci_upper = np.percentile(zcb_prices, 97.5)
        
        return {
            'mean_price': mean_price,
            'std_price': std_price,
            'ci_95': (ci_lower, ci_upper),
            'all_prices': zcb_prices,
            'pricing_time': actual_pricing_time
        }
    
    def compare_analytical_vs_mc(self, time_grid, paths, pricing_times, bond_maturities):
        results = {}
        
        for pricing_time in pricing_times:
            results[pricing_time] = {}
            
            for bond_maturity in bond_maturities:
                if bond_maturity <= pricing_time:
                    continue  # Skip if bond has already matured
                
                # Analytical price
                analytical_price = self.analytical_zcb_price(pricing_time, bond_maturity)
                
                # Monte Carlo price
                mc_result = self.mc_zcb_price_from_paths(
                    time_grid, paths, pricing_time, bond_maturity
                )
                
                results[pricing_time][bond_maturity] = {
                    'analytical': analytical_price,
                    'mc_mean': mc_result['mean_price'],
                    'mc_std': mc_result['std_price'],
                    'mc_ci': mc_result['ci_95'],
                    'difference': mc_result['mean_price'] - analytical_price
                }
        
        return results

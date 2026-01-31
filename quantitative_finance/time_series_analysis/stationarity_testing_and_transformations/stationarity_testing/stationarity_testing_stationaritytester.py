from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

class StationarityTester:
    """Comprehensive stationarity testing suite"""
    
    def __init__(self):
        self.critical_values_adf = {
            0.01: -3.43, 0.05: -2.86, 0.10: -2.57
        }
    
    def adf_test(self, y, regression='c', maxlag=None):
        """
        Augmented Dickey-Fuller test
        regression: 'n' (none), 'c' (constant), 'ct' (constant+trend)
        """
        result = adfuller(y, regression=regression, maxlag=maxlag, autolag='AIC')
        
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'lags': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
    
    def kpss_test(self, y, regression='c', nlags='auto'):
        """
        KPSS test: H0 is stationarity (opposite of ADF!)
        regression: 'c' (level), 'ct' (trend)
        """
        result = kpss(y, regression=regression, nlags=nlags)
        
        return {
            'kpss_stat': result[0],
            'p_value': result[1],
            'lags': result[2],
            'critical_values': result[3],
            'stationary': result[1] > 0.05  # Fail to reject H0
        }
    
    def phillips_perron(self, y, regression='c', lags=None):
        """
        Phillips-Perron test
        Non-parametric correction for serial correlation
        """
        n = len(y)
        
        # OLS regression
        if regression == 'c':
            X = np.ones((n, 1))
        elif regression == 'ct':
            X = np.column_stack([np.ones(n), np.arange(n)])
        else:
            X = None
        
        y_lag = y[:-1]
        dy = np.diff(y)
        
        if X is not None:
            X_reg = np.column_stack([X[1:], y_lag])
        else:
            X_reg = y_lag.reshape(-1, 1)
        
        model = OLS(dy, X_reg).fit()
        
        if regression == 'c':
            rho_idx = 1
        elif regression == 'ct':
            rho_idx = 2
        else:
            rho_idx = 0
        
        rho = model.params[rho_idx]
        se_rho = model.bse[rho_idx]
        
        # Long-run variance (Newey-West)
        if lags is None:
            lags = int(4 * (n/100)**(2/9))
        
        residuals = model.resid
        gamma0 = np.var(residuals)
        gamma_sum = 0
        
        for lag in range(1, lags+1):
            weight = 1 - lag / (lags + 1)  # Bartlett kernel
            gamma_lag = np.cov(residuals[:-lag], residuals[lag:])[0, 1]
            gamma_sum += 2 * weight * gamma_lag
        
        sigma2_lr = gamma0 + gamma_sum
        
        # PP adjustment
        lambda_hat = 0.5 * (sigma2_lr - gamma0)
        
        # Corrected test statistic
        se_correction = np.sqrt(sigma2_lr / gamma0)
        pp_stat = rho / (se_rho * se_correction) - (n * se_rho * lambda_hat) / (gamma0 * se_correction)
        
        # Use ADF critical values (approximately)
        p_value = None  # Would need to interpolate from tables
        stationary = pp_stat < -2.86  # Approximate 5% critical value
        
        return {
            'pp_stat': pp_stat,
            'p_value': p_value,
            'lags': lags,
            'stationary': stationary
        }
    
    def variance_ratio_test(self, y, lags=[2, 4, 8, 16]):
        """
        Variance ratio test (Lo-MacKinlay)
        H0: Random walk (VR=1)
        """
        n = len(y)
        returns = np.diff(y)
        
        mu = np.mean(returns)
        var1 = np.sum((returns - mu)**2) / (n - 1)
        
        results = []
        for q in lags:
            # q-period returns
            returns_q = np.array([np.sum(returns[i:i+q]) for i in range(0, n-q, q)])
            m = len(returns_q)
            
            varq = np.sum((returns_q - q*mu)**2) / m
            
            # Variance ratio
            vr = varq / (q * var1)
            
            # Asymptotic variance under H0 (i.i.d.)
            theta = 2 * (2*q - 1) * (q - 1) / (3 * q)
            asy_var_iid = theta / (n - q + 1)
            
            # Test statistic
            z_stat = (vr - 1) / np.sqrt(asy_var_iid)
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            
            results.append({
                'lag': q,
                'VR': vr,
                'z_stat': z_stat,
                'p_value': p_value,
                'reject_rw': p_value < 0.05
            })
        
        return results
    
    def zivot_andrews(self, y, model='C', trim=0.15):
        """
        Zivot-Andrews unit root test with one structural break
        model: 'A' (level), 'B' (trend), 'C' (both)
        """
        n = len(y)
        trim_n = int(n * trim)
        
        break_points = range(trim_n, n - trim_n)
        min_adf = np.inf
        opt_break = None
        
        for tb in break_points:
            # Create break dummies
            DU = np.zeros(n)
            DU[tb:] = 1
            
            DT = np.zeros(n)
            DT[tb:] = np.arange(n - tb)
            
            # Construct regression
            t_trend = np.arange(n)
            y_lag = y[:-1]
            dy = np.diff(y)
            
            if model == 'A':
                X = np.column_stack([np.ones(n-1), t_trend[1:], DU[1:], y_lag])
            elif model == 'B':
                X = np.column_stack([np.ones(n-1), t_trend[1:], DT[1:], y_lag])
            else:  # 'C'
                X = np.column_stack([np.ones(n-1), t_trend[1:], DU[1:], DT[1:], y_lag])
            
            # Add lagged differences (simplified: use 1 lag)
            dy_lag = np.diff(y, n=1)[:-1]
            X_full = np.column_stack([X[:-1], dy_lag])
            y_reg = dy[1:]
            
            try:
                model_fit = OLS(y_reg, X_full).fit()
                adf_stat = model_fit.tvalues[-2]  # Coefficient on y_lag
                
                if adf_stat < min_adf:
                    min_adf = adf_stat
                    opt_break = tb
            except:
                continue
        
        # ZA critical values more negative than standard ADF
        # 5% critical value approximately -4.8 for model C
        critical_value = -4.8 if model == 'C' else -4.5
        
        return {
            'za_stat': min_adf,
            'break_point': opt_break,
            'critical_value': critical_value,
            'stationary': min_adf < critical_value
        }
    
    def comprehensive_test(self, y, name='Series'):
        """Run all tests and summarize"""
        print(f"\n{'='*60}")
        print(f"STATIONARITY TESTS: {name}")
        print(f"{'='*60}")
        
        # ADF
        adf = self.adf_test(y, regression='c')
        print(f"\nAugmented Dickey-Fuller (H0: Unit Root):")
        print(f"  Statistic: {adf['adf_stat']:.4f}")
        print(f"  p-value: {adf['p_value']:.4f}")
        print(f"  Lags: {adf['lags']}")
        print(f"  Conclusion: {'Stationary' if adf['stationary'] else 'Non-stationary (unit root)'}")
        
        # KPSS
        kpss_result = self.kpss_test(y, regression='c')
        print(f"\nKPSS (H0: Stationary):")
        print(f"  Statistic: {kpss_result['kpss_stat']:.4f}")
        print(f"  p-value: {kpss_result['p_value']:.4f}")
        print(f"  Conclusion: {'Stationary' if kpss_result['stationary'] else 'Non-stationary'}")
        
        # Phillips-Perron
        pp = self.phillips_perron(y, regression='c')
        print(f"\nPhillips-Perron (H0: Unit Root):")
        print(f"  Statistic: {pp['pp_stat']:.4f}")
        print(f"  Lags: {pp['lags']}")
        print(f"  Conclusion: {'Stationary' if pp['stationary'] else 'Non-stationary'}")
        
        # Combined interpretation
        print(f"\n{'='*60}")
        if adf['stationary'] and kpss_result['stationary']:
            print("CONSENSUS: Series is STATIONARY")
        elif not adf['stationary'] and not kpss_result['stationary']:
            print("CONSENSUS: Series is NON-STATIONARY (unit root)")
        else:
            print("AMBIGUOUS: Tests disagree - possible near-unit-root or structural break")
        
        return {
            'adf': adf,
            'kpss': kpss_result,
            'pp': pp
        }

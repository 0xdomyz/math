import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline, griddata
import warnings

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("VOLATILITY SURFACE AND SKEW ANALYSIS")
print("="*60)

class VolatilitySurface:
    """Volatility surface construction and analysis"""
    
    def __init__(self, S0, r=0.05):
        self.S0 = S0
        self.r = r
        self.market_data = {}
    
    def add_market_data(self, K, T, market_price, option_type='call'):
        """Add market option data"""
        if T not in self.market_data:
            self.market_data[T] = []
        self.market_data[T].append({
            'K': K, 'price': market_price, 'type': option_type
        })
    
    def implied_vol_newton(self, K, T, market_price, option_type='call', 
                          initial_guess=0.2, tol=1e-6, max_iter=100):
        """Calculate implied volatility using Newton-Raphson"""
        sigma = initial_guess
        
        for i in range(max_iter):
            # BS price and vega
            d1 = (np.log(self.S0/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                price = self.S0*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
            else:
                price = K*np.exp(-self.r*T)*norm.cdf(-d2) - self.S0*norm.cdf(-d1)
            
            vega = self.S0 * norm.pdf(d1) * np.sqrt(T)
            
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            
            if vega < 1e-10:
                return np.nan
            
            sigma = sigma - diff / vega
            sigma = max(0.001, min(sigma, 5.0))
        
        return np.nan
    
    def build_iv_surface(self):
        """Build IV surface from market data"""
        surface = {}
        
        for T in sorted(self.market_data.keys()):
            surface[T] = []
            for option in self.market_data[T]:
                iv = self.implied_vol_newton(option['K'], T, option['price'], option['type'])
                if not np.isnan(iv):
                    moneyness = option['K'] / self.S0
                    log_moneyness = np.log(moneyness)
                    surface[T].append({
                        'K': option['K'],
                        'moneyness': moneyness,
                        'log_moneyness': log_moneyness,
                        'iv': iv
                    })
        
        return surface
    
    def check_butterfly_arbitrage(self, surface, T):
        """Check for butterfly arbitrage violations"""
        if T not in surface or len(surface[T]) < 3:
            return True, []
        
        violations = []
        points = sorted(surface[T], key=lambda x: x['K'])
        
        for i in range(1, len(points) - 1):
            K_low = points[i-1]['K']
            K_mid = points[i]['K']
            K_high = points[i+1]['K']
            
            # Approximate butterfly value from IVs
            # Should be non-negative for no arbitrage
            iv_low = points[i-1]['iv']
            iv_mid = points[i]['iv']
            iv_high = points[i+1]['iv']
            
            # Simplified check: convexity in variance
            w1 = (K_high - K_mid) / (K_high - K_low)
            w2 = (K_mid - K_low) / (K_high - K_low)
            
            iv_interp = w1 * iv_low + w2 * iv_high
            
            # If actual IV significantly above interpolated, may violate
            if iv_mid < iv_interp * 0.8:  # Threshold
                violations.append((K_mid, iv_mid, iv_interp))
        
        return len(violations) == 0, violations
    
    def check_calendar_arbitrage(self, surface):
        """Check for calendar arbitrage (variance must increase)"""
        maturities = sorted(surface.keys())
        violations = []
        
        if len(maturities) < 2:
            return True, []
        
        # Check at ATM
        atm_variances = []
        for T in maturities:
            points = surface[T]
            # Find closest to ATM
            atm_point = min(points, key=lambda x: abs(x['moneyness'] - 1.0))
            total_var = atm_point['iv']**2 * T
            atm_variances.append((T, total_var))
        
        for i in range(1, len(atm_variances)):
            if atm_variances[i][1] < atm_variances[i-1][1]:
                violations.append((atm_variances[i-1][0], atm_variances[i][0]))
        
        return len(violations) == 0, violations

class SVIModel:
    """SVI (Stochastic Volatility Inspired) volatility smile model"""
    
    @staticmethod
    def svi_variance(k, a, b, rho, m, sigma):
        """SVI raw parameterization: total variance as function of log-moneyness"""
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def svi_implied_vol(k, T, a, b, rho, m, sigma):
        """Convert SVI variance to implied volatility"""
        var = SVIModel.svi_variance(k, a, b, rho, m, sigma)
        return np.sqrt(var / T) if var > 0 else 0.01
    
    @staticmethod
    def calibrate(log_moneyness, ivs, T, initial_guess=None):
        """Calibrate SVI to market IVs"""
        if initial_guess is None:
            # Initial guess
            a_init = np.mean(ivs)**2 * T
            b_init = 0.1
            rho_init = -0.3
            m_init = 0.0
            sigma_init = 0.2
            initial_guess = [a_init, b_init, rho_init, m_init, sigma_init]
        
        target_variances = ivs**2 * T
        
        def objective(params):
            a, b, rho, m, sigma = params
            model_vars = np.array([SVIModel.svi_variance(k, a, b, rho, m, sigma) 
                                   for k in log_moneyness])
            return np.sum((model_vars - target_variances)**2)
        
        # Constraints for no arbitrage (simplified)
        bounds = [
            (0.001, None),      # a > 0
            (0.001, None),      # b > 0
            (-0.999, 0.999),    # -1 < rho < 1
            (None, None),       # m
            (0.001, None)       # sigma > 0
        ]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            return result.x, result.fun
        else:
            return initial_guess, np.inf

# Generate synthetic market with skew
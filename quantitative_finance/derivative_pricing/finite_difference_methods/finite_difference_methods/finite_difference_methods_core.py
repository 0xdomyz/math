import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("FINITE DIFFERENCE METHODS FOR OPTION PRICING")
print("="*70)

class FiniteDifferenceEngine:
    """Finite difference PDE solver for options"""
    
    def __init__(self, S0, K, r, sigma, T, option_type='call'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type
    
    def setup_grid(self, M, N, S_max_factor=3.0):
        """Create spatial and temporal grid"""
        # Space grid
        self.S_max = S_max_factor * self.K
        self.S_min = 0.0
        self.M = M  # Spatial points
        self.N = N  # Time steps
        
        self.dS = (self.S_max - self.S_min) / M
        self.dt = self.T / N
        
        self.S_grid = np.linspace(self.S_min, self.S_max, M+1)
        self.t_grid = np.linspace(0, self.T, N+1)
        
        # Grid for option values V[i, j] = V(S_i, t_j)
        self.V = np.zeros((M+1, N+1))
        
        # Terminal condition (payoff)
        if self.option_type == 'call':
            self.V[:, -1] = np.maximum(self.S_grid - self.K, 0)
        else:  # put
            self.V[:, -1] = np.maximum(self.K - self.S_grid, 0)
    
    def intrinsic_value(self):
        """Intrinsic value for American options"""
        if self.option_type == 'call':
            return np.maximum(self.S_grid - self.K, 0)
        else:
            return np.maximum(self.K - self.S_grid, 0)
    
    def apply_boundary_conditions(self, j):
        """Apply boundary conditions at time step j"""
        t = self.t_grid[j]
        tau = self.T - t
        
        if self.option_type == 'call':
            # S → 0: V → 0
            self.V[0, j] = 0
            # S → ∞: V → S - K*e^(-r*tau)
            self.V[-1, j] = self.S_max - self.K * np.exp(-self.r * tau)
        else:  # put
            # S → 0: V → K*e^(-r*tau)
            self.V[0, j] = self.K * np.exp(-self.r * tau)
            # S → ∞: V → 0
            self.V[-1, j] = 0
    
    def explicit_method(self, american=False):
        """Explicit finite difference (forward time, centered space)"""
        # Check stability condition
        max_dt = self.dS**2 / (self.sigma**2 * self.S_max**2)
        if self.dt > max_dt:
            print(f"Warning: Stability condition violated!")
            print(f"  Current dt: {self.dt:.6f}, Max stable dt: {max_dt:.6f}")
        
        # Coefficients for interior points
        alpha = np.zeros(self.M+1)
        beta = np.zeros(self.M+1)
        gamma = np.zeros(self.M+1)
        
        for i in range(1, self.M):
            S = self.S_grid[i]
            
            # Discretized PDE coefficients
            a = 0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 - self.r * S / self.dS)
            b = -self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r)
            c = 0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r * S / self.dS)
            
            alpha[i] = a
            beta[i] = 1 + b
            gamma[i] = c
        
        # Time stepping (backward from T to 0)
        for j in range(self.N-1, -1, -1):
            # Update interior points
            for i in range(1, self.M):
                self.V[i, j] = (alpha[i] * self.V[i-1, j+1] + 
                               beta[i] * self.V[i, j+1] + 
                               gamma[i] * self.V[i+1, j+1])
            
            # Apply boundary conditions
            self.apply_boundary_conditions(j)
            
            # American exercise
            if american:
                intrinsic = self.intrinsic_value()
                self.V[:, j] = np.maximum(self.V[:, j], intrinsic)
        
        return self.get_option_price()
    
    def implicit_method(self, american=False):
        """Implicit finite difference (backward time)"""
        # Build tridiagonal matrix
        diag_a = np.zeros(self.M-1)
        diag_b = np.zeros(self.M-1)
        diag_c = np.zeros(self.M-1)
        
        for idx, i in enumerate(range(1, self.M)):
            S = self.S_grid[i]
            
            # Coefficients
            a = -0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 - self.r * S / self.dS)
            b = 1 + self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r)
            c = -0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r * S / self.dS)
            
            diag_a[idx] = a
            diag_b[idx] = b
            diag_c[idx] = c
        
        # Create sparse tridiagonal matrix
        A = diags([diag_a[1:], diag_b, diag_c[:-1]], [-1, 0, 1], format='csr')
        
        # Time stepping
        for j in range(self.N-1, -1, -1):
            # Right-hand side
            rhs = self.V[1:-1, j+1].copy()
            
            # Adjust for boundary conditions at time j
            self.apply_boundary_conditions(j)
            rhs[0] -= diag_a[0] * self.V[0, j]
            rhs[-1] -= diag_c[-1] * self.V[-1, j]
            
            # Solve linear system
            self.V[1:-1, j] = spsolve(A, rhs)
            
            # American exercise
            if american:
                intrinsic = self.intrinsic_value()
                self.V[:, j] = np.maximum(self.V[:, j], intrinsic)
        
        return self.get_option_price()
    
    def crank_nicolson(self, american=False, rannacher_steps=4):
        """Crank-Nicolson (theta=0.5) with Rannacher smoothing"""
        # Coefficients for interior points
        alpha = np.zeros(self.M-1)
        beta = np.zeros(self.M-1)
        gamma = np.zeros(self.M-1)
        
        for idx, i in enumerate(range(1, self.M)):
            S = self.S_grid[i]
            
            a = 0.25 * self.dt * (self.sigma**2 * S**2 / self.dS**2 - self.r * S / self.dS)
            b = -0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r)
            c = 0.25 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r * S / self.dS)
            
            alpha[idx] = a
            beta[idx] = b
            gamma[idx] = c
        
        # LHS matrix: (I + 0.5*L)
        diag_a_lhs = -alpha
        diag_b_lhs = 1 - beta
        diag_c_lhs = -gamma
        
        A_lhs = diags([diag_a_lhs[1:], diag_b_lhs, diag_c_lhs[:-1]], 
                      [-1, 0, 1], format='csr')
        
        # Time stepping
        for j in range(self.N-1, -1, -1):
            # Rannacher smoothing: Use fully implicit for first few steps
            use_implicit = (j >= self.N - rannacher_steps)
            
            if use_implicit:
                theta = 1.0  # Fully implicit
            else:
                theta = 0.5  # Crank-Nicolson
            
            # RHS matrix depends on theta
            if theta == 0.5:
                # Standard Crank-Nicolson
                rhs = (alpha * self.V[:-2, j+1] + 
                       (1 + beta) * self.V[1:-1, j+1] + 
                       gamma * self.V[2:, j+1])
            else:
                # Fully implicit
                rhs = self.V[1:-1, j+1].copy()
            
            # Boundary conditions
            self.apply_boundary_conditions(j)
            rhs[0] -= diag_a_lhs[0] * self.V[0, j]
            rhs[-1] -= diag_c_lhs[-1] * self.V[-1, j]
            
            # Solve
            self.V[1:-1, j] = spsolve(A_lhs, rhs)
            
            # American exercise
            if american:
                intrinsic = self.intrinsic_value()
                self.V[:, j] = np.maximum(self.V[:, j], intrinsic)
        
        return self.get_option_price()
    
    def get_option_price(self):
        """Interpolate to get price at S0"""
        # Find closest grid point to S0
        idx = np.argmin(np.abs(self.S_grid - self.S0))
        return self.V[idx, 0]
    
    def compute_greeks(self):
        """Compute Greeks from grid at t=0"""
        idx = np.argmin(np.abs(self.S_grid - self.S0))
        
        if idx == 0 or idx == self.M:
            return {'delta': None, 'gamma': None, 'theta': None}
        
        # Delta: ∂V/∂S
        delta = (self.V[idx+1, 0] - self.V[idx-1, 0]) / (2 * self.dS)
        
        # Gamma: ∂²V/∂S²
        gamma = (self.V[idx+1, 0] - 2*self.V[idx, 0] + self.V[idx-1, 0]) / self.dS**2
        
        # Theta: ∂V/∂t (use first time step)
        if self.N > 0:
            theta = (self.V[idx, 1] - self.V[idx, 0]) / self.dt
        else:
            theta = None
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta}

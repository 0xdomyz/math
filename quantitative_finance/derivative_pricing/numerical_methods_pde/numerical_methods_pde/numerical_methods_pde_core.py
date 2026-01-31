import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
    from scipy.stats import norm

# Block 1

# =====================================
# FINITE DIFFERENCE PDE SOLVERS
# =====================================
print("="*70)
print("PDE METHODS FOR OPTION PRICING")
print("="*70)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Analytical Black-Scholes price for comparison."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def explicit_fd_european(S0, K, T, r, sigma, option_type='put', M=100, N=1000):
    """
    Explicit finite difference method for European options.
    
    Conditionally stable: Requires Δt ≤ Δx²/σ²S_max²
    """
    # Grid parameters
    S_max = 3 * K  # Maximum stock price
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    # Stock price grid
    S = np.linspace(S_min, S_max, M+1)
    
    # Initialize option values at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:  # put
        V = np.maximum(K - S, 0)
    
    # Stability check
    max_dt = dS**2 / (sigma**2 * S_max**2)
    if dt > max_dt:
        print(f"   WARNING: Explicit scheme unstable! dt={dt:.6f} > max_dt={max_dt:.6f}")
    
    # Backward time-stepping
    for n in range(N):
        V_new = V.copy()
        
        # Interior points (i=1 to M-1)
        for i in range(1, M):
            # Coefficients from PDE discretization
            a = 0.5 * dt * (r * i - sigma**2 * i**2)
            b = 1 - dt * (sigma**2 * i**2 + r)
            c = 0.5 * dt * (r * i + sigma**2 * i**2)
            
            V_new[i] = a*V[i-1] + b*V[i] + c*V[i+1]
        
        # Boundary conditions
        if option_type == 'call':
            V_new[0] = 0  # V(0,t) = 0 for call
            V_new[M] = S_max - K*np.exp(-r*(T - (n+1)*dt))  # V(S_max,t) ≈ S_max
        else:  # put
            V_new[0] = K*np.exp(-r*(T - (n+1)*dt))  # V(0,t) = Ke^(-r(T-t))
            V_new[M] = 0  # V(S_max,t) = 0 for put
        
        V = V_new
    
    # Interpolate to S0
    option_price = np.interp(S0, S, V)
    
    return option_price, S, V

def implicit_fd_european(S0, K, T, r, sigma, option_type='put', M=100, N=500):
    """
    Implicit finite difference method for European options.
    
    Unconditionally stable, requires solving tridiagonal system.
    """
    S_max = 3 * K
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    S = np.linspace(S_min, S_max, M+1)
    
    # Initialize at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Construct tridiagonal matrix
    alpha = np.zeros(M+1)
    beta = np.zeros(M+1)
    gamma = np.zeros(M+1)
    
    for i in range(1, M):
        alpha[i] = -0.5 * dt * (r * i - sigma**2 * i**2)
        beta[i] = 1 + dt * (sigma**2 * i**2 + r)
        gamma[i] = -0.5 * dt * (r * i + sigma**2 * i**2)
    
    # Create tridiagonal matrix
    diagonals = [alpha[1:M], beta[1:M], gamma[1:M-1]]
    A = diags(diagonals, [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Backward time-stepping
    for n in range(N):
        b = V[1:M].copy()
        
        # Adjust for boundary conditions
        if option_type == 'call':
            V[0] = 0
            V[M] = S_max - K*np.exp(-r*(T - (n+1)*dt))
        else:
            V[0] = K*np.exp(-r*(T - (n+1)*dt))
            V[M] = 0
        
        b[0] -= alpha[1] * V[0]
        b[-1] -= gamma[M-1] * V[M]
        
        # Solve tridiagonal system
        V[1:M] = spsolve(A, b)
    
    option_price = np.interp(S0, S, V)
    return option_price, S, V

def crank_nicolson_european(S0, K, T, r, sigma, option_type='put', M=100, N=500):
    """
    Crank-Nicolson method for European options (θ=0.5).
    
    Second-order accurate in time, unconditionally stable.
    """
    S_max = 3 * K
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    S = np.linspace(S_min, S_max, M+1)
    
    # Initialize at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Coefficients
    alpha = np.zeros(M+1)
    beta = np.zeros(M+1)
    gamma = np.zeros(M+1)
    
    for i in range(1, M):
        alpha[i] = 0.25 * dt * (r * i - sigma**2 * i**2)
        beta[i] = -0.5 * dt * (sigma**2 * i**2 + r)
        gamma[i] = 0.25 * dt * (r * i + sigma**2 * i**2)
    
    # LHS matrix (implicit part)
    A_diags = [-alpha[1:M], 1-beta[1:M], -gamma[1:M-1]]
    A = diags(A_diags, [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Backward time-stepping
    for n in range(N):
        # RHS (explicit part)
        b = np.zeros(M-1)
        for i in range(1, M):
            b[i-1] = alpha[i]*V[i-1] + (1+beta[i])*V[i] + gamma[i]*V[i+1]
        
        # Boundary conditions
        if option_type == 'call':
            V[0] = 0
            V[M] = S_max - K*np.exp(-r*(T - (n+1)*dt))
        else:
            V[0] = K*np.exp(-r*(T - (n+1)*dt))
            V[M] = 0
        
        b[0] -= alpha[1] * V[0]
        b[-1] -= gamma[M-1] * V[M]
        
        # Solve system
        V[1:M] = spsolve(A, b)
    
    option_price = np.interp(S0, S, V)
    return option_price, S, V

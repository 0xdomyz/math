import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

# Auto-extracted from markdown file
# Source: basel_accords.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("BASEL ACCORDS: CAPITAL REQUIREMENTS CALCULATION")
print("="*70)

class BaselCapitalCalculator:
    """Calculate regulatory capital under Basel frameworks"""
    
    def __init__(self):
        self.minimum_cet1 = 0.045  # 4.5%
        self.ccb = 0.025  # Capital conservation buffer
        self.minimum_total = 0.08  # 8%
        
    def basel_i_rwa(self, exposure, asset_class='corporate'):
        """Basel I risk-weighted assets (crude buckets)"""
        weights = {
            'sovereign_oecd': 0.0,
            'bank_oecd': 0.2,
            'mortgage': 0.5,
            'corporate': 1.0,
            'other': 1.0
        }
        
        rw = weights.get(asset_class, 1.0)
        return exposure * rw
    
    def basel_ii_standardized_rwa(self, exposure, rating='unrated', asset_class='corporate'):
        """Basel II Standardized Approach"""
        # Corporate risk weights by external rating
        corporate_weights = {
            'AAA': 0.20, 'AA': 0.20,
            'A': 0.50,
            'BBB': 1.00,
            'BB': 1.00,
            'B': 1.50,
            'CCC': 1.50,
            'unrated': 1.00
        }
        
        # Asset class specific
        if asset_class == 'corporate':
            rw = corporate_weights.get(rating, 1.0)
        elif asset_class == 'retail':
            rw = 0.75
        elif asset_class == 'mortgage':
            rw = 0.35
        elif asset_class == 'sovereign':
            sovereign_weights = {'AAA': 0.0, 'AA': 0.0, 'A': 0.20, 
                               'BBB': 0.50, 'BB': 1.00, 'B': 1.00, 'unrated': 1.00}
            rw = sovereign_weights.get(rating, 1.0)
        else:
            rw = 1.0
        
        return exposure * rw
    
    def basel_ii_irb_correlation(self, pd, asset_class='corporate'):
        """Asset correlation R in Basel IRB formula"""
        if asset_class == 'corporate':
            # Corporate correlation formula
            R = (0.12 * (1 - np.exp(-50*pd))/(1 - np.exp(-50)) +
                 0.24 * (1 - (1 - np.exp(-50*pd))/(1 - np.exp(-50))))
        elif asset_class == 'retail':
            # Retail (simplified)
            R = 0.15
        else:
            R = 0.12
        
        return R
    
    def basel_ii_irb_rw(self, pd, lgd, ead, maturity=2.5, asset_class='corporate'):
        """
        Basel II IRB risk weight formula
        
        Parameters:
        - pd: Probability of default (annual)
        - lgd: Loss given default (as fraction)
        - ead: Exposure at default
        - maturity: Effective maturity in years
        """
        # Floor for PD
        pd = max(pd, 0.0003)  # 0.03% floor
        
        # Correlation
        R = self.basel_ii_irb_correlation(pd, asset_class)
        
        # Maturity adjustment factor
        b = (0.11852 - 0.05478 * np.log(pd))**2
        
        # Basel IRB formula
        # RW = LGD × N[(1-R)^(-0.5) × G(PD) + (R/(1-R))^0.5 × G(0.999)] × [1 + (M-2.5)×b]/(1-1.5×b) × 1.06
        
        # Calculate components
        G_pd = norm.ppf(pd)  # Inverse normal of PD
        G_999 = norm.ppf(0.999)  # 99.9th percentile
        
        # Capital requirement (before LGD scaling)
        K = norm.cdf(
            ((1-R)**(-0.5)) * G_pd + ((R/(1-R))**0.5) * G_999
        )
        
        # Maturity adjustment
        if asset_class == 'corporate':
            MA = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        else:
            MA = 1.0  # No maturity adjustment for retail
        
        # Risk weight
        RW = lgd * K * MA * 1.06  # 1.06 is scaling factor
        
        # RWA
        rwa = ead * RW
        
        return rwa, RW
    
    def calculate_capital_ratios(self, rwa, cet1_capital, tier1_capital, total_capital):
        """Calculate capital ratios"""
        cet1_ratio = cet1_capital / rwa if rwa > 0 else 0
        tier1_ratio = tier1_capital / rwa if rwa > 0 else 0
        total_ratio = total_capital / rwa if rwa > 0 else 0
        
        return {
            'cet1_ratio': cet1_ratio,
            'tier1_ratio': tier1_ratio,
            'total_ratio': total_ratio
        }
    
    def check_compliance(self, cet1_ratio, ccyb=0.0, gsib_buffer=0.0):
        """Check Basel III compliance"""
        required_cet1 = self.minimum_cet1 + self.ccb + ccyb + gsib_buffer
        
        compliant = cet1_ratio >= required_cet1
        
        return {
            'required_cet1': required_cet1,
            'actual_cet1': cet1_ratio,
            'excess': cet1_ratio - required_cet1,
            'compliant': compliant
        }

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

warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("STATIONARITY TESTING AND TRANSFORMATIONS")
print("="*80)

from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("CLASSICAL DECOMPOSITION AND FILTERING: TIME SERIES ANALYSIS")
print("="*80)

from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import warnings

from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FREQUENCY DOMAIN AND SPECTRAL ANALYSIS")
print("="*80)

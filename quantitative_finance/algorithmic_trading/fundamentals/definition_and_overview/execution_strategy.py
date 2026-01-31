import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class ExecutionStrategy:
    """Base class for execution algorithms."""
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        raise NotImplementedError

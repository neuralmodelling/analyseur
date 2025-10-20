# ~/analyseur/cbgt/parameters.py
#
# Documentation by Lungsi 20 Oct 2025
#
# This contains function for loading the files
#

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SimulationParams:
    duration: float = 10000 # ms
    t_start_recording = 2000 # ms
    dt: float = 0.1 # ms
    _1000ms: int = 1000
    significant_digits: int = 3
    nuclei_ctx: List[str] = None
    nuclei_bg: List[str] = None
    # nuclei_thal: List[str] = None

    def __post_init__(self):
        if self.nuclei_ctx is None:
            self.nuclei_ctx = ["CSN", "PTN", "IN"]
        if self.nuclei_bg is None:
            self.nuclei_bg = ["FSI", "GPe", "GPi", "MSN", "STN"]


@dataclass
class SpikeAnalysisParams:
    window: Tuple[float, float] = (0, 5)
    binsz: float = 0.05

    def validate(self):
        if self.binsz <= 0:
            raise ValueError("bin size must be positive")
        if self.window[1] <= self.window[0]:
            raise ValueError("time window end must be greater than start")

#custom_param = SpikeAnalysisParams(window=(0,5), binsz=0.02)
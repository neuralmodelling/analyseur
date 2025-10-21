# ~/analyseur/cbgt/parameters.py
#
# Documentation by Lungsi 20 Oct 2025
#
# This contains function for loading the files
#

from dataclasses import dataclass, field
from typing import List, Tuple

DEFAULT_CONDUCTANCES = {
    "cortex": {
        "g_L": 0.05, # This is called g_L_mean in CBGTC and unit is mS.cm-2
    },
    "bg": {},
    "thalamus": {
        "g_L": 0.05, # This is called g_L_mean in CBGTC and unit is mS.cm-2
    }
}

@dataclass
class SimulationParams:
    duration: float = 10000 # ms
    t_start_recording = 2000 # ms
    dt: float = 0.1 # ms
    _1000ms: int = 1000
    significant_digits: int = 3
    significant_digits_ephys: int = 5  # very small values for disinhibition experiments
    nuclei_ctx: List[str] = None
    nuclei_bg: List[str] = None
    # nuclei_thal: List[str] = None
    neurotrans: List[str] = None
    conductance: dict = field(default_factory=lambda: DEFAULT_CONDUCTANCES.copy())

    def __post_init__(self):
        if self.nuclei_ctx is None:
            self.nuclei_ctx = ["CSN", "PTN", "IN"]
        if self.nuclei_bg is None:
            self.nuclei_bg = ["FSI", "GPe", "GPi", "MSN", "STN"]
        if self.neurotrans is None:
            self.neurotrans = ['AMPA', 'NMDA', 'GABAA', 'GABAB']


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
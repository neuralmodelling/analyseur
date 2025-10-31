# ~/analyseur/cbgt/parameters.py
#
# Documentation by Lungsi 20 Oct 2025
#
# This contains function for loading the files
#

from dataclasses import dataclass, field
from typing import List, Tuple
import math

DEFAULT_CONDUCTANCES = {
    "cortex": { # 0.05
        "g_L": 1.0, # This is called g_L_mean in CBGTC and unit is mS.cm-2
    },
    "bg": {},
    "thalamus": {
        "g_L": 0.05, # This is called g_L_mean in CBGTC and unit is mS.cm-2
    },
}

DEFAULT_FEEDFORWORD_CURRENTS = {
    "cortex": {
        "CSN": 1.124,
        "PTN": 1.25,
        "IN": 1.178,
    },
    "bg": {},
    "thalamus": {},
}

def bin_size_by_rule(total_time=None, rule=None, frequency=None):
    """
    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    match rule:
        case "Square Root":
            n_bins = round( math.sqrt(total_time) )
        case "Rice Rule":
            n_bins = round( 2 * math.cbrt(total_time) )
        case "Periodic":
            period = 1 / frequency

    if rule in ["Square Root", "Rice Rule"]:
        return total_time / n_bins
    else:
        return 2 * period  # two periods in one bin

@dataclass
class SimulationParams:
    """
    ================
    SimulationParams
    ================

    Default parameters from CBGT simulation

    +----------------------+---------------------------------------+
    | Parameter name       | Value                                 |
    +======================+=======================================+
    | `duration`           | `10000` milliseconds                  |
    +----------------------+---------------------------------------+
    | `t_start_recording`  | `2000` milliseconds                   |
    +----------------------+---------------------------------------+
    | `dt`                 | `0.1` milliseconds                    |
    +----------------------+---------------------------------------+
    | `nuclei_ctx`         | `["CSN", "PTN", "IN"]`                |
    +----------------------+---------------------------------------+
    | `nuclei_bg`          | `["FSI", "GPe", "GPi", "MSN", "STN"]` |
    +----------------------+---------------------------------------+
    | `nuclei_thal`        | `["TRN", "MD"]`                       |
    +----------------------+---------------------------------------+
    | `neurotrans`         | `['AMPA', 'NMDA', 'GABAA', 'GABAB']`  |
    +----------------------+---------------------------------------+
    | `conductance`        | `DEFAULT_CONDUCTANCES`                |
    +----------------------+---------------------------------------+
    | `ff_currents`        | `DEFAULT_FEEDFORWORD_CURRENTS`        |
    +----------------------+---------------------------------------+

    --------
    Use Case
    --------
    ::
        from analyseur.cbgt.parameters import SimulationParams

        simparams = SimulationParams()


    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    duration: float = 10000 # ms
    t_start_recording = 2000 # ms
    dt: float = 0.1 # ms
    nuclei_ctx: List[str] = None
    nuclei_bg: List[str] = None
    nuclei_thal: List[str] = None
    neurotrans: List[str] = None
    conductance: dict = field(default_factory=lambda: DEFAULT_CONDUCTANCES.copy())
    ff_currents: dict = field(default_factory=lambda: DEFAULT_FEEDFORWORD_CURRENTS.copy())

    def __post_init__(self):
        if self.nuclei_ctx is None:
            self.nuclei_ctx = ["CSN", "PTN", "IN"]
        if self.nuclei_bg is None:
            self.nuclei_bg = ["FSI", "GPe", "GPi", "MSN", "STN"]
        if self.neurotrans is None:
            self.neurotrans = ['AMPA', 'NMDA', 'GABAA', 'GABAB']

@dataclass
class SignalAnalysisParams:
    """
    ====================
    SignalAnalysisParams
    ====================

    Default parameters for signal analysis

    +------------------------+---------------------+
    | Parameter name         | Value               |
    +========================+=====================+
    | `_1000ms`              | `1000` milliseconds |
    +------------------------+---------------------+
    | `decimal_places`       | `3`                 |
    +------------------------+---------------------+
    | `decimal_places_ephys` | `5`                 |
    +------------------------+---------------------+
    | `window`               | `(0, 10)` seconds   |
    +------------------------+---------------------+
    | `sampling_period`      | `0.0001` seconds    |
    +------------------------+---------------------+
    | `sampling_period_ms`   | `0.1` milliseconds  |
    +------------------------+---------------------+
    | `binsz_sqrt_rule`      | `100`               |
    +------------------------+---------------------+
    | `binsz_rice_rule`      | `232.558`           |
    +------------------------+---------------------+
    | `binsz_10perbin`       | `0.001`             |
    +------------------------+---------------------+
    | `binsz_100perbin`      | `0.01`              |
    +------------------------+---------------------+
    | `binsz_1000perbin`     | `0.1`               |
    +------------------------+---------------------+

    --------
    Use Case
    --------
    ::
        from analyseur.cbgt.parameters import SignalAnalysisParams

        siganal = SignalAnalysisParams()


    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    _1000ms: int = 1000
    decimal_places: int = 3
    decimal_places_ephys: int = 5  # very small values for disinhibition experiments

    window: Tuple[float, float] = (0, SimulationParams.duration / _1000ms)
    sampling_period_ms: float = SimulationParams.dt
    sampling_period: float = SimulationParams.dt / _1000ms

    binsz_sqrt_rule: float = bin_size_by_rule(SimulationParams.duration, "Square Root")
    binsz_rice_rule: float = bin_size_by_rule(SimulationParams.duration, "Rice Rule")
    binsz_10perbin: float = 10 * sampling_period
    binsz_100perbin: float = 100 * sampling_period
    binsz_1000perbin: float = 1000 * sampling_period

    std_Gaussian_kernel: float = 2

    def validate(self):
        if any(binsz <= 0 for binsz in [binsz_sqrt_rule, binsz_rice_rule,
                                        binsz_10perbin, binsz_100perbin,
                                        binsz_1000perbin]):
            raise ValueError("bin size must be positive")
        if self.window[1] <= self.window[0]:
            raise ValueError("time window end must be greater than start")

#custom_param = SpikeAnalysisParams(window=(0,5), binsz=0.02)
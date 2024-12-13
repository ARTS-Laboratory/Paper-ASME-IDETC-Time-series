from dataclasses import dataclass


@dataclass
class NormalInverseGamma:
    alpha: float
    beta: float
    mu: float
    kappa: float


@dataclass
class NormalInverseGammaRunLength:
    run_length: int
    params: NormalInverseGamma
    # alpha: float
    # beta: float
    # mu: float
    # kappa: float

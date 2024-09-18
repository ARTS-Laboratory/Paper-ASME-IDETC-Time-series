from dataclasses import dataclass


@dataclass
class NormalInverseGamma:
    alpha: float
    beta: float
    mu: float
    kappa: float


@dataclass
class NormalInverseGammaRunLength:
    alpha: float
    beta: float
    mu: float
    kappa: float
    run_length: int

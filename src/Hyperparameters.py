from abc import ABC
from dataclasses import dataclass


@dataclass
class BOCPDHyperparams:
    alpha: float
    beta: float
    mu: float
    kappa: float
    lamb: float


@dataclass
class CUSUMHyperparams:
    mean: float
    std_dev: float
    h: float
    alpha: float


@dataclass
class EMHyperparams:
    normal_data_size: int
    abnormal_data_size: int
    normal_mean: float
    abnormal_mean: float
    normal_var: float
    abnormal_var: float
    pi: float
    epochs: int


@dataclass
class GreyHyperparams:
    window_size: int
    critical_value: float
    critical_ratio_value: float
    alpha: float


@dataclass
class NonparametricHyperparams:
    window_size: int
    critical_value: float
    alpha: float

from dataclasses import dataclass
from enum import Enum
from os import PathLike


class ModelType(Enum):
    BOCPD = 'bocpd'
    EM = 'expectation maximization'
    CUSUM = 'cusum'
    GREY_MODEL = 'grey'
    NON_PARAMETRIC = 'nonparametric'


@dataclass
class DetectionAlgorithm:
    type: ModelType
    name: str
    with_progress: bool = False
    save_path: PathLike = None
    hyperparameters: dataclass = None
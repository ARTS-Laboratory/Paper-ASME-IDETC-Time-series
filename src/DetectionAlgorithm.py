from dataclasses import dataclass


@dataclass
class DetectionAlgorithm:
    name: str
    with_progress: bool = False
    save_path: str = None
    hyperparameters: dataclass = None
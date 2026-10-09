"""Fixed main-experiment settings."""
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data" / "BCICIV_4_mat"
CACHE_DIR = HERE / "cache_hybrid44"
SEED = 42
FULL_EPOCHS = {1: 112, 2: 69, 3: 104}
SCORED_FINGERS = [0, 1, 2, 4]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
# Fixed training-derived electrode sets used in the completed hybrid_4 experiment.
FULL_ELECTRODES = {
    1: [0, 4, 7, 14, 16, 22, 23, 30, 32, 38, 39, 41, 42, 47, 48, 57],
    2: [1, 2, 5, 7, 11, 13, 14, 17, 18, 21, 22, 23, 25, 30, 36, 39],
    3: [8, 12, 17, 22, 23, 37, 40, 42, 48, 51, 52, 53, 55, 56, 57, 58],
}


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    window: int = 512
    inference_window: int = 2048
    batch: int = 64
    windows_per_epoch: int = 8192
    lr: float = 8.42e-5
    weight_decay: float = 1e-4
    dropout: float = 0.1
    channel_count: int = 16
    features: int = 44
    max_epochs: int = 100
    patience: int = 25

    def __post_init__(self):
        if (self.seed, self.window, self.inference_window) != (42, 512, 2048):
            raise ValueError("The main experiment requires seed 42 and windows 512/2048.")
        if (self.channel_count, self.features) != (16, 44):
            raise ValueError("The main experiment requires 16 electrodes and 44 features.")

    def to_dict(self):
        return asdict(self)

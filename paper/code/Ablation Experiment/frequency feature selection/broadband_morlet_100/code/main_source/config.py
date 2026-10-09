"""Fixed main-experiment settings."""
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data" / "BCICIV_4_mat"
SEED = 42
FULL_EPOCHS = {1: 112, 2: 69, 3: 104}
SCORED_FINGERS = [0, 1, 2, 4]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


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
    features: int = 43
    max_epochs: int = 100
    patience: int = 25

    def __post_init__(self):
        if (self.seed, self.window, self.inference_window) != (42, 512, 2048):
            raise ValueError("The main experiment requires seed 42 and windows 512/2048.")

    def to_dict(self):
        return asdict(self)

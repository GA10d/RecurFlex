"""Use an unmodified snapshot of the current Main Experiment implementation."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".plotcache"))
sys.path.insert(0, str(HERE / "main_source"))
from config import Config
from data import raw_features, labels100, fit_statistics, transform, normalize_targets, inverse_targets, sample_batch
from model import RecurFlex, training_model
from inference import predict, restore_1000hz, coverage
from metrics import score
from train import objective
from utils import setup, save_json, load_json, sha256


def protocol():
    return load_json(HERE / "protocol.json")


def record(subject):
    return next(r for r in protocol()["records"] if r["subject"] == subject)

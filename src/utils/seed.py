"""Global seeding for reproducible runs."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (when installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN kernels; slightly slower but reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

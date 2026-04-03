import time
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from collections import defaultdict


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


_timers = defaultdict(float)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    _timers[title] += elapsed
    print(f"  {title} — {elapsed:.3f}s")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

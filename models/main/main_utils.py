import torch
import torch.nn.functional as F
from train_utils import ce_loss

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


class LinearWarmupScalar:
    """
    Simple linear ramp from start -> end within warmup_iters iterations.
    If warmup_iters == 0 the value is fixed at `end`.
    """

    def __init__(self, start, end, warmup_iters):
        self.start = float(start)
        self.end = float(end)
        self.warmup_iters = max(1, int(warmup_iters))

    def get_value(self, iteration):
        progress = min(1.0, float(iteration) / float(self.warmup_iters))
        return self.start + (self.end - self.start) * progress

    def __call__(self, iteration):
        return self.get_value(iteration)

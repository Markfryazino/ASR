import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PolarityInversion(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PolarityInversion(*args, **kwargs, sample_rate=16000)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

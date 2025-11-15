from weakref import proxy

import torch


class InitializationResults:
    def __init__(self, a: torch.Tensor):
        self._a = a

    @property
    def a(self) -> torch.Tensor:
        return proxy(self._a)

     # self.mask_a_init, self.c_init, self.b_init, self.superpixel_dict

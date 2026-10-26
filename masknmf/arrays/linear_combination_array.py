from collections.abc import Iterable, Sequence

import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike

class LinearCombinationArray(ArrayLike):
    """
    Lazy linear combination of same-shaped arrays: ``self[item] = sum_i alphas[i] * arrays[i][item]``.

    The default alphas are all 1, giving the sum of the arrays. Set them explicitly for anything else, e.g. a residual:
    ``LinearCombinationArray([pmd, fluctuating_background, signals, static_background], alphas=[1, -1, -1, -1])``.

    No device, dtype or return-type checking is done: each component is indexed with the same ``item`` and the
    results are combined with ``*`` and ``+``, so the components must return values that broadcast and add together
    (e.g. all torch tensors on one device). Indexing follows the ``ArrayLike`` conventions.
    """

    def __init__(self, arrays: Sequence[ArrayLike], alphas: Iterable | None = None):
        """
        Args:
            arrays (Sequence[ArrayLike]): The arrays to combine. All must have the same shape.
            alphas (Iterable | None): One scalar per array (python numbers, numpy scalars or 0-d torch tensors).
                Defaults to all 1s, i.e. the sum of the arrays.
        """
        self._arrays = list(arrays)
        if len(self._arrays) == 0:
            raise ValueError("LinearCombinationArray needs at least one array")

        if alphas is None:
            alphas = [1.0] * len(self._arrays)
        self._alphas = list(alphas)
        if len(self._alphas) != len(self._arrays):
            raise ValueError(f"Got {len(self._alphas)} alphas for {len(self._arrays)} arrays")

        self._shape = tuple(self._arrays[0].shape)
        for k, arr in enumerate(self._arrays[1:], start=1):
            if tuple(arr.shape) != self._shape:
                raise ValueError(f"Array {k} has shape {tuple(arr.shape)}, expected {self._shape}")

    @property
    def arrays(self) -> list[ArrayLike]:
        return self._arrays

    @property
    def alphas(self) -> list:
        return self._alphas

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self):
        return self._arrays[0].dtype

    def __getitem__(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ):
        # The first product is a new tensor, never a view of a component (static arrays return
        # stride-0 views), so accumulating into it in place is safe.
        cumulator = self._alphas[0] * self._arrays[0][item]
        for alpha, arr in zip(self._alphas[1:], self._arrays[1:]):
            cumulator += alpha * arr[item]
        return cumulator

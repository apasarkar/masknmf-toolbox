"""Per-cell statistics a user brings along (a custom ordering, a score per neuron) to sort demixed signals by."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CellStats:
    """
    One row per cell, one column per named stat.

    Args:
        names (tuple[str, ...]): one name per column
        values (np.ndarray): shape (num_cells, num_stats), cast to float32
    """

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"cell stats must be (num_cells, num_stats), got {values.shape}")
        names = tuple(str(n) for n in self.names)
        if len(names) != values.shape[1]:
            raise ValueError(f"{len(names)} names for {values.shape[1]} stat columns")
        if len(set(names)) != len(names):
            raise ValueError(f"stat names repeat: {names}")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)

    @classmethod
    def read(cls, path) -> "CellStats":
        """
        Load stats from a file, by suffix:

        - ``.npy``: a (num_cells,) array named after the file, a (num_cells, num_stats) array whose
          columns are named ``<file> 0``, ``<file> 1``, ..., or a structured array whose numeric fields
          are the stats
        - ``.npz``: one (num_cells,) array per stat, named by key
        - ``.csv`` / ``.tsv``: a header row of names, then one row per cell
        """
        path = Path(path)
        match path.suffix.lower():
            case ".npy":
                values = np.load(path)
                if values.dtype.names:
                    names = [n for n in values.dtype.names if values[n].dtype.kind in "biuf"]
                    values = np.column_stack([values[n] for n in names])
                else:
                    names = [path.stem] if values.ndim == 1 else [f"{path.stem} {j}" for j in range(values.shape[1])]
            case ".npz":
                with np.load(path) as f:
                    names = list(f.keys())
                    values = np.column_stack([f[n] for n in names]) if names else np.zeros((0, 0))
            case ".csv" | ".tsv":
                table = np.genfromtxt(path, delimiter="," if path.suffix.lower() == ".csv" else "\t", names=True)
                names = list(table.dtype.names)
                values = np.column_stack([np.atleast_1d(table[n]) for n in names])
            case suffix:
                raise ValueError(f"unsupported cell stats file {suffix!r}: use .npy, .npz, .csv or .tsv")
        return cls(tuple(names), values)

    @classmethod
    def from_results(cls, results) -> "CellStats":
        """
        Per-signal stats from what the results already hold. Of the demixed trace ``c``: mean, std, snr
        (peak over std) and skew. Against the compressed movie averaged over the footprint
        (``pmd_roi_averages``): fit, the trace's correlation with it; resid and bkgd, the residual's and the
        fluctuating background's std over that average's.
        """
        c = np.asarray(results.temporal_demixed.detach().cpu(), dtype=np.float32)
        mean = c.mean(0)
        std = c.std(0) + 1e-6
        skew = ((c - mean) ** 3).mean(0) / std**3
        pmd = np.asarray(results.pmd_roi_averages.detach().cpu(), dtype=np.float32)
        resid = np.asarray(results.residual_roi_averages.detach().cpu(), dtype=np.float32)
        bkgd = np.asarray(results.fluctuating_background_roi_averages.detach().cpu(), dtype=np.float32)
        pmd_std = pmd.std(1) + 1e-6
        fit = ((c.T - mean[:, None]) * (pmd - pmd.mean(1, keepdims=True))).mean(1) / (std * pmd_std)
        return cls(
            ("mean", "std", "snr", "skew", "fit", "resid", "bkgd"),
            np.column_stack([mean, std, c.max(0) / std, skew, fit, resid.std(1) / pmd_std, bkgd.std(1) / pmd_std]),
        )

    @classmethod
    def from_order(cls, order, num_cells: int, name: str = "order") -> "CellStats":
        """Ranks from a custom cell order: cell ``order[i]`` gets rank ``i``; cells left out get NaN and sort last."""
        order = np.asarray(order)
        if order.dtype.names is not None or order.dtype.kind not in "iuf":
            raise ValueError("a cell order is a plain array of signal ids; a stats table goes in as cell_stats")
        order = order.astype(np.int64).ravel()
        if len(np.unique(order)) != len(order) or (len(order) and (order.min() < 0 or order.max() >= num_cells)):
            raise ValueError(f"a cell order lists distinct cell ids below {num_cells}")
        ranks = np.full(num_cells, np.nan, np.float32)
        ranks[order] = np.arange(len(order))
        return cls((name,), ranks)

    def join(self, other: "CellStats") -> "CellStats":
        """Both sets of columns over the same cells; ``other``'s replace same-named ones."""
        if other.values.shape[0] != self.values.shape[0]:
            raise ValueError(f"{other.values.shape[0]} rows joined onto {self.values.shape[0]}")
        keep = [i for i, n in enumerate(self.names) if n not in other.names]
        return CellStats(tuple(self.names[i] for i in keep) + other.names, np.column_stack([self.values[:, keep], other.values]))

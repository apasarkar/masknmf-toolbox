# PMD Decomposition

Efficient Penalized Matrix Decomposition (PMD) for compressing and denoising large-scale functional imaging datasets (e.g., calcium imaging or voltage imaging).

`pmd_decomposition()` automatically divides data into overlapping spatial blocks, performs truncated randomized local SVD to get low-rank spatial and temporal decompositions, filters out noisy components, and stitches results together into a global clean, low-rank representation of your data ready for analysis.

---

## Table of Contents

- [What does pmd_decomposition() do?](#what-does-pmd_decomposition-do)
- [Key Features](#key-features)
- [Input Parameters](#input-parameters)
- [Output](#output)
- [Example Usage](#example-usage)
- [Notes for Developers](#notes-for-developers)
- [Credits](#credits)
- [API Reference — pmd_decomposition()](#api-reference-—-pmd_decomposition)

---

## What does pmd_decomposition() do?

`pmd_decomposition()` performs a blockwise localized low-rank factorization of your dataset:

\[
\text{data} \approx U \times V
\]

- **U** — spatial basis matrix (sparse, `[H×W, n_components]`)  
- **V** — temporal basis matrix (dense, `[n_components, T]`)  

Together they represent a compressed and denoised form of your movie.

---

## Key Features

- **Block-wise processing:** Splits large FOVs into overlapping spatial blocks for efficient local analysis.  
- **Automatic calibration:** Simulates random noise to set smoothness (roughness) thresholds automatically before decomposition.  
- **Localized low-rank decomposition:** Runs fast truncated randomized SVDs on each block to find local low-rank signal bases.  
- **Optional neural denoising:** Can apply spatial or temporal neural-network-based denoisers before or after decomposition.  
- **Adaptive component selection:** Each block automatically chooses its rank by rejecting components that fail smoothness tests.  
- **Smooth merging:** Combines block results into a global sparse representation (`U, V`) with pixelwise mean and variance estimates.  
- **Compact and fast:** Outputs a lightweight `PMDArray` object for rapid downstream demixing or neural signal extraction.  
- **GPU-ready:** Fully compatible with PyTorch, enabling real-time or near-real-time processing.  

---

## Input Parameters

| Argument | Type | Description |
|----------|------|------------|
| `dataset` | `np.ndarray` or `torch.Tensor` | Input movie with shape `(T, H, W)` |
| `block_sizes` | `(int, int)` | Spatial block size used for local decomposition |
| `frame_range` | `int` | Number of frames to process (subset of total frames) |
| `max_components` | `int` | Maximum number of components per block |
| `device` | `"cpu"` or `"cuda"` | Compute device |
| `(optional) spatial_denoiser, temporal_denoiser` | `Callable` | Optional denoiser from previous neural network training |

---

## Output

Returns a `PMDArray` object containing:

- `u` → spatial basis (sparse COO tensor)  
- `v` → temporal basis (dense tensor)  
- `mean_img`, `var_img` → per-pixel statistics  
- `u_local_projector` → (optional) local spatial projector for fast re-projection  

Example reconstruction:

```python
reconstructed = (pmd_result.u.to_dense() @ pmd_result.v).reshape(T, H, W)
```

## Example Usage

```python
import numpy as np
from masknmf.compression import pmd_decomposition

# Example dataset
data = (np.random.rand(200, 64, 64) * 255).astype(np.float32)

result = pmd_decomposition(
    dataset=data,
    block_sizes=(16, 16),
    frame_range=150,
    max_components=10,
    device="cuda"
)

U = result.u.to_dense()
V = result.v
print("U:", U.shape, "V:", V.shape)
```

## Notes for Developers

This implementation omits explicit spatial/temporal penalties from the original PMD (Buchanan et al., 2018) and instead uses roughness-based thresholds to reject noisy components.

A randomized SVD algorithm (Halko et al., 2011) is used for efficient local decomposition.

Each block adaptively selects its rank based on smoothness statistics, stopping when components become too noisy.

Use .coalesce() to merge duplicate indices in sparse COO tensors.

## Credits

Adapted from Buchanan et al., 2018 (bioRxiv) and used in the maskNMF pipeline (Pasarkar et al., 2023).

Built with PyTorch for flexible CPU/GPU performance.

## API Reference — pmd_decomposition()
| Parameter                  | Type                                             | Default | Description                                                                      |
| -------------------------- | ------------------------------------------------ | ------- | -------------------------------------------------------------------------------- |
| `dataset`                  | `masknmf.ArrayLike` or `masknmf.LazyFrameLoader` | —       | Input dataset of shape `(frames, height, width)`                                 |
| `block_sizes`              | `Tuple[int, int]`                                | —       | Spatial block size for local decomposition. Each dimension must be ≥10.          |
| `frame_range`              | `int`                                            | —       | Number of frames used to estimate spatial and temporal bases.                    |
| `max_components`           | `int`                                            | 20      | Maximum number of components per spatial block.                                  |
| `sim_conf`                 | `int`                                            | 5       | Percentile value defining roughness thresholds for keeping/rejecting components. |
| `frame_batch_size`         | `int`                                            | 10000   | Max frames loaded into memory at one time.                                       |
| `max_consecutive_failures` | `int`                                            | 1       | Stops accepting new components after this many failures.                         |
| `spatial_avg_factor`       | `int`                                            | 1       | Optional spatial downsampling factor.                                            |
| `temporal_avg_factor`      | `int`                                            | 1       | Optional temporal downsampling factor.                                           |
| `compute_normalizer`       | `bool`                                           | True    | If True, estimates per-pixel noise variance (`var_img`).                         |
| `pixel_weighting`          | `Optional[np.ndarray]`                           | None    | Optional spatial weighting map `(H, W)` to upweight signal pixels.               |
| `spatial_denoiser`         | `Optional[torch.nn.Module]`                      | None    | Optional callable applied to spatial components.                                 |
| `temporal_denoiser`        | `Optional[torch.nn.Module]`                      | None    | Optional callable applied to temporal traces.                                    |
| `device`                   | `str`                                            | `"cpu"` | Compute device: `"cpu"` or `"cuda"`.                                             |
---

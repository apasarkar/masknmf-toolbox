# GlutamateCalciumSpinePipeline

```python
import tifffile
from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline

glu = tifffile.imread("green.tif")
calcium = tifffile.imread("red.tif")

pipe = GlutamateCalciumSpinePipeline(output_folder="results/")
pipe.run(glutamate_channel=glu, calcium_channel=calcium)
```

Writes `results/<YYYYMMDD_HHMMSS>_glutamate-calcium-spine/results.<channel>.hdf5`, one file per channel.
- Either channel may be `None`; its file is then absent.
- `run` drops the first 200 frames (`exclude_initial_frames`);
- every output has `frames - 200` frames.
- the `retained_frames` dataset lists the raw frame indices that were kept; index the raw stack with it.

```
results.glutamate.hdf5, results.calcium.hdf5
  retained_frames
  RigidRegistrationArray, RigidMotionCorrector
  CompressionArray
  DemixingResults           spines
  global/DemixingResults    whole-dendrite component
```

`masknmf.utils.results_files(run)` maps channel to file: `{"calcium": run / "results.calcium.hdf5", ...}`.

```python
from pathlib import Path
import h5py
import numpy as np
import masknmf

run = Path("results/20260908_112158_glutamate-calcium-spine")
results = masknmf.utils.results_files(run)["calcium"]
with h5py.File(results, "r") as f:
    retained_frames = f["retained_frames"][()]
calcium = tifffile.imread("red.tif")[retained_frames]
timings = np.arange(calcium.shape[0]) / 19.66  # optional seconds axis
```

## 1. Motion correction

| hdf5 groups |
|---|
| `RigidRegistrationArray` (shifts, sinc_margin), `RigidMotionCorrector` (template, max_shifts, batch_size) |

The registered movie is recreated from the raw frames and the stored shifts.

```python
reg = masknmf.RigidRegistrationArray.from_hdf5(results, input_movie=calcium)
```

| attribute | shape / type |
|---|---|
| `reg.shifts` | `(frames, 2)` tensor, `(dy, dx)` px |
| `reg.strategy.template` | `(H, W)` ndarray |
| `reg[i]` | `(1, H, W)` tensor, registered frame |
| `reg[:]` | `(frames, H, W)` tensor |

```python
masknmf.MotionCorrectionVis(reg, frame_timings=timings, mean_subtract=True).show()
```

## 2. Compression

| hdf5 groups |
|---|
| `CompressionArray` (spatial_compressed, temporal_compressed, mean_image, noise_variance_image, spatial_compressed_local_projector, shape) |

```python
pmd = masknmf.CompressionArray.from_hdf5(results)
```

| attribute | shape / type |
|---|---|
| `pmd.shape` | `(frames, H, W)` |
| `pmd.spatial_compressed`, `pmd.temporal_compressed` | `(H*W, rank)` sparse, `(rank, frames)` |
| `pmd.mean_image`, `pmd.noise_variance_image` | `(H, W)` tensors |
| `pmd[i]` | `(1, H, W)` ndarray, raw scale |

Compression ran on the registered movie masked to the dendrite (`mean_image == 0` outside).

```python
moco = reg[:].cpu().numpy()
masknmf.CompressionVis(moco, pmd, frame_timings=timings, device="cuda").show()
```

## 3. Demixing

| file / hdf5 group | contents |
|---|---|
| `results.glutamate.hdf5` `DemixingResults` | spines found in glutamate |
| `results.calcium.hdf5` `DemixingResults` | same footprints, traces refit on calcium |
| either file, `global/DemixingResults` | one whole-dendrite component |

```python
res = masknmf.DemixingResults.from_hdf5(results, device="cuda")
global_res = masknmf.DemixingResults.from_hdf5(results, prefix="global", device="cuda")
```

| attribute | shape / type |
|---|---|
| `res.a` | `(H*W, K)` sparse footprints |
| `res.signals_array.export_a()` | `(H, W, K)` ndarray |
| `res.c` | `(frames, K)` traces |
| `res.mean_image` | `(H, W)` |
| `res.compression_array`, `res.signals_array`, `res.residual_array` | lazy `(frames, H, W)` movies |

```python
masknmf.SingleSessionDemixingVis(res, frame_timings=timings, device="cuda").show()
```

## Scripts

`fpl.loop.run()` after `show()` outside notebooks. `examples/pipelines/run_glu_ca_pipeline.py` runs,
`examples/pipelines/view_glu_ca_results.py` views.

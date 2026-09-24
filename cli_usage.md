# masknmf command line

## Subcommands

| command | what it does |
|---|---|
| `masknmf pipelines` | list the pipelines and their config sections |
| `masknmf params --pipeline P` | list every parameter P accepts, grouped by section and kind |
| `masknmf run --pipeline P ...` | build P from flags and call `P.run(...)` |
| `masknmf run --pipeline P --help` | argparse help showing only P's flags |
| `masknmf view RESULTS.hdf5 ...` | open the viewers for the stages a results file holds |

```bash
masknmf pipelines
masknmf params --pipeline two-photon-calcium
masknmf run --pipeline two-photon-calcium --help
```

Pipelines: `two-photon-calcium`, `one-photon-culture`, `glutamate-calcium-spine`, `widefield-singlechannel`.

## How flags map to the Python API

| Python | CLI |
|---|---|
| `run(data=...)` (the only movie argument) | positional `MOVIE` (optional when `data` accepts `None`) |
| `run(glutamate_channel=..., calcium_channel=...)` (two movies) | `--glutamate-channel PATH --calcium-channel PATH` |
| `run(x: np.ndarray)` | `--x PATH.npy` |
| `run(frame_rate=30)` | `--frame-rate 30` or `--fs 30` |
| `run(scalar=...)` / `__init__(scalar=...)` | `--scalar VALUE` (underscores become dashes) |
| `__init__(motion_correct_config=PiecewiseRigidMotionCorrectionConfig(...))` | `--motion-correct-kind piecewise-rigid` |
| `__init__(motion_correct_config="skip")` | `--motion-correct-kind skip` |
| a field inside that config | `--set motion-correct.max_rigid_shifts=10,10` |

Parsing:
- tuples are comma separated: `--set compress.block_sizes=32,32`
- booleans take `true/false/1/0/yes/no/on/off`: `--remove-intermediates false`
- optional values take `none`: `--set compress.frame_range=none`
- `--set` can be repeated. It can be given without `--<section>-kind` only when the section has a single real kind. Otherwise it errors and asks for the kind.
- fields marked `*` in `masknmf params` (arrays, templates, detrenders, pixel/frame weighting) cannot be set from the CLI.

Movie input (`MOVIE`, `--glutamate-channel`, `--calcium-channel`, `view --raw`):
- `movie.tif` / `movie.tiff`: `TiffArray`
- a directory: every `.tif`/`.tiff` in it, sorted by name, as a `TiffSeriesLoader` (`TiffArray` if there is only one)
- `movie.h5` / `movie.hdf5`: `Hdf5Array`, which requires `--dataset NAME`

---

## two-photon-calcium (`TwoPhotonCalciumPipeline`)

Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise | skip), `spatial-highpass` (spatial-highpass), `filtered-demixing` / `unfiltered-demixing` (multipass, not CLI-buildable, so defaults always apply).
Run args: `MOVIE` (optional), `--fs` (required), `--exclude-border-radius`, `--remove-intermediates`.
Init args: `--output-folder`, `--frame-batch-size`, `--device {auto,cuda,cpu}`.

```bash
# minimal: all defaults (rigid moco, default compression, default demixing) -> ./<timestamp>_two-photon-calcium/results.hdf5
masknmf run --pipeline two-photon-calcium movie.tif --fs 30

# directory of tiffs, run folder under ./session1_out, keep intermediate groups
masknmf run --pipeline two-photon-calcium ./session1_tiffs/ --fs 30 \
    --output-folder ./session1_out \
    --remove-intermediates false

# hdf5 input
masknmf run --pipeline two-photon-calcium raw.h5 --dataset /mov --fs 15

# rigid moco with a smaller search window
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 \
    --motion-correct-kind rigid --set motion-correct.max_shifts=10,10

# piecewise-rigid moco, every field set
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 \
    --motion-correct-kind piecewise-rigid \
    --set motion-correct.minimum_patch_sizes=64,64 \
    --set motion-correct.overlaps=8,8 \
    --set motion-correct.max_rigid_shifts=15,15 \
    --set motion-correct.max_deviation_rigid=3,3

# already registered data: skip moco, zero the border of the shift mask
masknmf run --pipeline two-photon-calcium registered.tif --fs 30 \
    --motion-correct-kind skip --exclude-border-radius 10

# plain PMD compression, tuned
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 \
    --compress-kind compress \
    --set compress.block_sizes=32,32 \
    --set compress.max_components=30 \
    --set compress.frame_range=5000 \
    --set compress.spatial_avg_factor=2 --set compress.temporal_avg_factor=2 \
    --set compress.compute_normalizer=false

# compression with denoising
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 \
    --compress-kind compress-denoise \
    --set compress.noise_variance_quantile=0.5 --set compress.num_epochs=20

# stronger spatial highpass before the filtered demixing pass
# (--set alone is enough because spatial-highpass has a single kind)
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 \
    --set spatial-highpass.filter_sigma=6

# resume from an existing run folder's compression, re-running only demixing into the same results.hdf5 (no movie needed)
masknmf run --pipeline two-photon-calcium --fs 30 --compress-kind skip \
    --output-folder ./20260923_120000_two-photon-calcium

# force CPU, smaller batches
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --device cpu --frame-batch-size 100
```

## one-photon-culture (`OnePhotonCulturePipeline`)

Sections: `motion-correct` (skip only; omit it to get gradient moco), `compress` (compress | compress-denoise | skip), `demixing` (multipass, not CLI-buildable).
Run args: `MOVIE`, `--fs` (required), `--indicator-sign {negative,positive}` (required), `--active-frames FILE.npy` (required), `--remove-intermediates`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device`.

`--active-frames` is a 1-D `.npy` with one 0/1 entry per frame. It is used as the compression frame weighting.

```bash
# minimal: gradient moco, default compression + demixing
masknmf run --pipeline one-photon-culture voltage.tif --fs 400 \
    --indicator-sign negative --active-frames active.npy

# positive indicator, no moco, data loaded into RAM
masknmf run --pipeline one-photon-culture voltage.tif --fs 1000 \
    --indicator-sign positive --active-frames active.npy \
    --motion-correct-kind skip --load-into-ram true

# denoised compression with larger blocks, keep the compression group
masknmf run --pipeline one-photon-culture voltage.h5 --dataset data --fs 400 \
    --indicator-sign negative --active-frames active.npy \
    --compress-kind compress-denoise --set compress.block_sizes=40,40 --set compress.max_components=40 \
    --output-folder ./voltage_out --remove-intermediates false

# re-demix an existing run folder's compression
masknmf run --pipeline one-photon-culture voltage.tif --fs 400 \
    --indicator-sign negative --active-frames active.npy --compress-kind skip \
    --output-folder ./voltage_out/20260923_120000_one-photon-culture
```

## glutamate-calcium-spine (`GlutamateCalciumSpinePipeline`)

Two movie arguments, so both are flags and neither is positional.
Sections: `motion-correct` (rigid), `compress` (compress-denoise), `demixing` (multipass, not CLI-buildable). Each has one kind, so `--set` works on its own.
Run args: `--glutamate-channel`, `--calcium-channel`, `--exclude-initial-frames` (default 200).
Init args: `--output-folder`, `--frame-batch-size`, `--device`.

```bash
# both channels
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --calcium-channel ca.tif

# glutamate only, results in a folder
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --output-folder ./spines_out

# calcium only, keep every frame
masknmf run --pipeline glutamate-calcium-spine \
    --calcium-channel ca.tif --exclude-initial-frames 0

# tune moco and compression without naming a kind
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --calcium-channel ca.tif \
    --set motion-correct.max_shifts=8,8 \
    --set compress.block_sizes=16,16 --set compress.num_epochs=5
```

## widefield-singlechannel (`WidefieldSinglechannelPipeline`)

Moco and compression only, with no demixing.
Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise, no skip).
Run args: `MOVIE`, `--exclude-border-radius`.
Init args: `--output-folder`, `--frame-batch-size`, `--device`.

```bash
# minimal
masknmf run --pipeline widefield-singlechannel wf.tif

# piecewise-rigid + plain compression (verified end to end on a 400x64x64 synthetic movie)
masknmf run --pipeline widefield-singlechannel wf.tif --device cpu \
    --output-folder ./wf_out \
    --motion-correct-kind piecewise-rigid --set motion-correct.minimum_patch_sizes=32,32 \
    --compress-kind compress --set compress.block_sizes=16,16

# zero a 10 px border of the compression pixel weighting
masknmf run --pipeline widefield-singlechannel wf.tif --exclude-border-radius 10

# no moco, denoised compression
masknmf run --pipeline widefield-singlechannel wf.tif \
    --motion-correct-kind skip --compress-kind compress-denoise --set compress.max_components=50
```

## view

```bash
masknmf view results.hdf5 --list                  # print which stage groups the file holds
masknmf view results.hdf5                         # demixing (or compression-only) viewer
masknmf view results.hdf5 --raw movie.tif --fs 30 # + motion and compression viewers (need the raw movie)
masknmf view results.hdf5 --raw raw.h5 --dataset /mov --device cpu
```

# masknmf command line

## Subcommands

| command | what it does |
|---|---|
| `masknmf` | open the launcher window: an overview, then pages that build and run a `masknmf run`, `view` or `track` command |
| `masknmf pipelines` | list the pipelines and their config sections |
| `masknmf params --pipeline P` | list every parameter P accepts and the value it uses by default |
| `masknmf params --pipeline P --json` | print P's default configs as a `--config` file |
| `masknmf run --pipeline P ...` | build P from flags and a `--config` file, and call `P.run(...)` |
| `masknmf run --pipeline P --help` | argparse help showing only P's flags |
| `masknmf view RESULTS.hdf5 ...` | open the viewer for the stages a results file holds |
| `masknmf view RESULTS... --classify` | label ROIs across one or more results files or globs |
| `masknmf train-classifier RESULTS... --out NAME` | train a ROI classifier on the saved labels |
| `masknmf classify RESULTS... --classifier F` | classify the ROIs in results files with a trained classifier |
| `masknmf track RESULTS... --out FOLDER` | track ROIs across sessions with ROICaT, adding a run to a tracking folder |
| `masknmf view TRACKING_FOLDER [RESULTS...]` | open the multisession viewer on a tracking folder's newest run |
| `masknmf view MANIFEST.json [RESULTS...]` | open the multisession viewer on the run a manifest names |
| `masknmf --version` | print the masknmf version |

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
| `__init__(motion_correct_config=PiecewiseRigidMotionCorrectionConfig())` | `--motion-correct-kind piecewise-rigid` |
| `__init__(motion_correct_config="skip")` | `--motion-correct-kind skip` |
| any field of any config, including each pass of a multipass config | `--config configs.json` |

- Run folder goes next to the movie (inside it for a directory of tiffs), in the working directory with no movie, or under `--output-folder`
- Config arguments left out take the pipeline's `default_configs()`, as printed by `masknmf params`
- `--<section>-kind` naming the default kind keeps the pipeline's defaults; another kind gets that config's own defaults

Parsing:
- booleans take `true/false/1/0/yes/no/on/off`: `--remove-intermediates true`
- fields marked `*` in `masknmf params` (arrays, templates, detrenders, pixel/frame weighting) are Python only

Movie input (`MOVIE`, `--glutamate-channel`, `--calcium-channel`, `view --raw`):
- `movie.tif` / `movie.tiff`: `TiffArray`
- a directory: every `.tif`/`.tiff` in it, sorted by name, as a `TiffSeriesLoader` (`TiffArray` if there is only one)
- `movie.h5` / `movie.hdf5`: `Hdf5Array`, which requires `--dataset NAME`

## Config files

`--config FILE` is json mapping `__init__` argument names to values.

- Each config is an object with a `"kind"` (`rigid`, `piecewise-rigid`, `compress`, `compress-denoise`, `multipass`, ...), or the string `"skip"`
- Only list what changes; everything else keeps the pipeline's default
- Switching a section's kind starts from that config's own defaults
- A list of passes replaces the default list; each pass builds on the default pass at the same position, extra passes on the last one
- Flags override the file: `--device cpu` replaces the file's device; a `--<section>-kind` naming a different kind resets that section to its defaults

```bash
# every default, ready to edit
masknmf params --pipeline two-photon-calcium --json > configs.json
masknmf run movie.tif --fs 30 --config configs.json   # the file names the pipeline, so --pipeline is optional

# rerun with an earlier run's configs (and frame rate, unless --fs is given)
masknmf run movie.tif --config ./20260923T120000_two-photon-calcium/config.json

# many movies, one run folder beside each; quote the glob
masknmf run "D:/sessions/*/movie.tif" --config ./20260923T120000_two-photon-calcium/config.json
```

Batches:
- several movies, or a glob, run one after another
- a movie that cannot be opened stops the batch before any run starts
- the file's `output_folder` is ignored when movies are given; each run folder goes beside its movie, or under `--output-folder`

```json
{
  "pipeline": "TwoPhotonCalciumPipeline",
  "configs": {
    "motion_correct_config": {"kind": "piecewise-rigid", "minimum_patch_sizes": [64, 64], "max_deviation_rigid": [3, 3]},
    "compress_config": {"kind": "compress-denoise", "max_components": 30},
    "filtered_demixing_config": {
      "kind": "multipass",
      "DemixingConfigs": [
        {"InitConfig": {"mad_correlation_threshold": 0.7}},
        {},
        {"NMFConfig": {"maxiter": 60}}
      ]
    },
    "device": "cuda"
  }
}
```

A run folder's `config.json`:
- `timings`: each step's start time, seconds, whether it finished, and peak cuda memory (GB) on cuda; updated as the run goes
- a step still running after 10 minutes is logged, and again every 10 minutes
- a resumed run (`--resume-from`, or `--compress-kind skip --output-folder <run folder>`) is a new run folder: the registration, and the compression it reuses, are copied in, the earlier folder is left as it is, and config.json keeps the earlier moco and compression configs, inputs and timings
- a run that fails keeps its folder, marked failed in config.json with the error in its log, whenever it saved a registration or a compression; from Python too
- `inputs`: the movie and arrays the run read, with their size and modification time; `masknmf run --config <run folder>/config.json` reruns on them unless a movie is given
- a finished run prints the `masknmf view` line that opens it, with the movie as `--raw`

---

## two-photon-calcium (`TwoPhotonCalciumPipeline`)

Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise | skip), `spatial-highpass` (spatial-highpass), `filtered-demixing` (multipass: 2 passes by default | skip), `unfiltered-demixing` (multipass: 3 passes by default).
Run args: `MOVIE` (optional), `--fs` (required), `--exclude-border-radius`, `--remove-intermediates`, `--stop-after {registration,compression,demixing}`, `--resume-from RESULTS`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device {auto,cuda,cpu}`, `--log-level {debug,info,warning}`.

- `--load-into-ram true` reads the raw movie into RAM before moco; needs about the movie's size in free RAM
- demixing passes without a detrender get a spline detrender built from `--fs`
- `--stop-after registration` writes the shifts and template and ends; `--stop-after compression` (or the older `--filtered-demixing-kind skip`) ends once the compression is written; a later `--compress-kind skip` run demixes it
- `--resume-from <results.hdf5>` copies that run's registration into the new run folder and replays it on the movie instead of estimating one, so compression can be re-run with other settings; with `--compress-kind skip` its compression is reused too, and demixing gets the registered movie back for raw traces
- `--remove-intermediates true` drops the compression once demixing is done; it is kept by default, so it can be checked

```bash
# all defaults -> <movie's folder>/<timestamp>_two-photon-calcium/results.hdf5
masknmf run --pipeline two-photon-calcium movie.tif --fs 30

# directory of tiffs, run folder under ./session1_out, drop the compression once demixed
masknmf run --pipeline two-photon-calcium ./session1_tiffs/ --fs 30 \
    --output-folder ./session1_out \
    --remove-intermediates true

# hdf5 input
masknmf run --pipeline two-photon-calcium raw.h5 --dataset /mov --fs 15

# piecewise-rigid moco at its defaults
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --motion-correct-kind piecewise-rigid

# registration only, then check it: raw beside registered with the shifts
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --stop-after registration
masknmf view ./20260923T120000_two-photon-calcium --raw movie.tif

# the same registration, compression re-run with other settings
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --stop-after compression \
    --resume-from ./20260923T120000_two-photon-calcium/results.hdf5 --compress-kind compress

# the run again, exactly as recorded
masknmf run --config ./20260923T120000_two-photon-calcium/config.json

# already registered: skip moco, zero the border of the shift mask
masknmf run --pipeline two-photon-calcium registered.tif --fs 30 \
    --motion-correct-kind skip --exclude-border-radius 10

# re-run only demixing on an earlier run's compression, in a new run folder beside it (no movie needed, no raw traces)
masknmf run --pipeline two-photon-calcium --fs 30 --compress-kind skip \
    --output-folder ./20260923T120000_two-photon-calcium

# the same with the movie, so demixing gets raw traces
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --compress-kind skip \
    --resume-from ./20260923T120000_two-photon-calcium/results.hdf5

# force CPU, smaller batches
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --device cpu --frame-batch-size 100
```

## one-photon-culture (`OnePhotonCulturePipeline`)

Sections: `motion-correct` (gradient | skip), `compress` (compress | compress-denoise | skip), `demixing` (multipass: 2 passes by default, no detrending).
Run args: `MOVIE`, `--fs` (required), `--indicator-sign {negative,positive}` (required), `--active-frames FILE.npy` (required), `--remove-intermediates`, `--stop-after {registration,compression,demixing}`, `--resume-from RESULTS`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device`, `--log-level`.

- `--active-frames`: 1-D `.npy`, one 0/1 per frame, used as the compression frame weighting
- gradient moco registers to the mean of the first `num_frames_template` frames (default 300)

```bash
# minimal: gradient moco, default compression + demixing
masknmf run --pipeline one-photon-culture voltage.tif --fs 400 \
    --indicator-sign negative --active-frames active.npy

# positive indicator, no moco, data loaded into RAM
masknmf run --pipeline one-photon-culture voltage.tif --fs 1000 \
    --indicator-sign positive --active-frames active.npy \
    --motion-correct-kind skip --load-into-ram true

# re-demix an earlier run's compression, in a new run folder beside it
masknmf run --pipeline one-photon-culture voltage.tif --fs 400 \
    --indicator-sign negative --active-frames active.npy --compress-kind skip \
    --output-folder ./voltage_out/20260923T120000_one-photon-culture
```

## glutamate-calcium-spine (`GlutamateCalciumSpinePipeline`)

Sections: `motion-correct` (rigid), `compress` (compress-denoise), `demixing` (multipass: 2 passes tuned for spines by default).
Run args: `--glutamate-channel`, `--calcium-channel`, `--exclude-initial-frames` (default 200), `--stop-after {registration,compression,demixing}`.
Init args: `--output-folder`, `--frame-batch-size`, `--device`, `--log-level`.

- both movies are flags, and either can be left out (`masknmf params` lists both as required)
- writes `results.glutamate.hdf5` and `results.calcium.hdf5` instead of `results.hdf5`
- always loads into RAM; there is no `--load-into-ram`

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
```

## widefield-singlechannel (`WidefieldSinglechannelPipeline`)

Moco and compression only, no demixing.
Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise, no skip).
Run args: `MOVIE`, `--exclude-border-radius`, `--stop-after {registration,compression}`, `--resume-from RESULTS`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device`, `--log-level`.

```bash
# minimal
masknmf run --pipeline widefield-singlechannel wf.tif

# zero a 10 px border of the compression pixel weighting
masknmf run --pipeline widefield-singlechannel wf.tif --exclude-border-radius 10

# no moco, plain compression at its defaults
masknmf run --pipeline widefield-singlechannel wf.tif --motion-correct-kind skip --compress-kind compress
```

## view

- `RESULTS`: one or more `.h5`/`.hdf5` files or quoted globs; `**` recurses
- globs skip `.zarr` stores, `.labels.hdf5` sidecars, and results files replaced by a newer curated file ([curated results files](#curated-results-files))
- the demixing viewer opens one file; several need `--classify`

A folder opens by the first match:

| the folder holds | opens |
|---|---|
| a `*_roicat-tracking-manifest.json`, directly or in its `tracking` subfolder | the multisession viewer on the newest run ([multisession](#multisession)) |
| `*results*.hdf5` files | as the glob `FOLDER/*results*.hdf5`: each results file's newest curated file, else the results file |

```bash
masknmf view "sessions/*/results.hdf5" --list      # print which stage groups each file holds
masknmf view results.hdf5                          # demixing viewer (compression only when there is no demixing)
masknmf view ./20260923T120000_two-photon-calcium   # the run folder's results.hdf5, or its newest curated file
masknmf view results.hdf5 --raw movie.tif --fs 30  # + raw and registered panels; a registration-only file needs --raw
masknmf view results.hdf5 --raw movie.tif --compression  # + lag-1 autocorrelation images of the registered, compressed and residual movies
masknmf view results.hdf5 --raw raw.h5 --dataset /mov --device cpu
masknmf view results.glutamate.hdf5 --prefix global  # the glutamate pipeline's whole-dendrite result
```

## classification

Label ROIs by hand, train a ROICaT classifier, classify new sessions.

- each results file with top-level demixing results is one session; others are skipped
- labels and predictions go beside each results file: `results.hdf5 -> results.labels.hdf5`
- `--classify` takes neither `--prefix`, `--raw` nor `--compression`

```bash
# label ROIs; --classifier is where Train saves, and an existing file is selected for Classify
masknmf view "sessions/*/results.hdf5" --classify --labels soma,dendrite,junk --classifier cells.roicat_classifier

# train; every ROI in every session must be labeled. writes cells.roicat_classifier and cells.training.json
masknmf train-classifier "sessions/*/results.hdf5" --out cells

# classify new sessions; unlabeled ROIs take the predictions
masknmf classify "new_sessions/**/results.hdf5" --classifier cells.roicat_classifier --device cpu
```

## multisession

`masknmf track` matches ROIs across sessions with ROICaT.

- one session per results file, numbered in the order given (a glob sorts by path)
- files without top-level demixing results are skipped; at least two sessions must remain

```bash
# --um-per-pixel is the imaging resolution (default 1.2); --device cpu keeps ROICaT off the gpu
masknmf track day1/results.hdf5 day2/results.hdf5 day3/results.hdf5 --out ./tracking
masknmf track "sessions/day*/*/results*.hdf5" --out ./tracking --um-per-pixel 0.8 --device cpu
```

`--out` is the experiment's tracking folder. Each run adds a timestamped folder and manifest, so one experiment can hold several runs:

```
experiment/
  day1/<run folder>/results.hdf5
  day2/<run folder>/results.hdf5
  tracking/
    20261001T180415_roicat-tracking/
    20261001T180415_roicat-tracking-manifest.json
    20261003T091240_roicat-tracking/
    20261003T091240_roicat-tracking-manifest.json
```

- run folder: `<timestamp>.tracking.results_all.richfile.zip`, `.run_data.richfile.zip`, `.params.yaml`, `.masknmf_sessions.json`
- manifest: what `masknmf view` opens, found by its `_roicat-tracking-manifest.json` name

```json
{
    "tracking": "20261001T180415_roicat-tracking",
    "sessions": ["../day1/<run folder>/results.hdf5", "../day2/<run folder>/results.hdf5"]
}
```

- `tracking` is the run folder, `sessions` the results files in session order
- paths are relative to the manifest with forward slashes, so the experiment survives being moved or opened on another OS
- a results file on another drive is recorded by absolute path
- keep the `.richfile.zip` files zipped: unzipping `run_data` on Windows silently drops files past the 260 character path limit, and a zip beside its unzipped copy is refused

From Python: `RoicatTracker().run_tracking(RoicatDataAdapter.from_masknmf(files))` then `.to_roicat_dir(folder)`;
`RoicatTrackingResults.from_manifest(path)` loads a run back. The launcher's Track sessions page builds the same command.

```bash
# newest run of a tracking folder
masknmf view ./tracking

# the same, from the folder holding the tracking folder
masknmf view .

# one run of several
masknmf view ./tracking/20261001T180415_roicat-tracking-manifest.json

# results files the manifest no longer finds: pass them after it
masknmf view ./tracking "sessions/day*/*/results.hdf5"

# print the sessions and the file each one reads, marking missing ones
masknmf view ./tracking "sessions/day*/*/results.hdf5" --list
```

- results files after the folder replace the recorded ones, each matched to its session by ROI count
- two sessions with the same ROI count need the files in session order
- a file whose ROI count differs from its session's is refused: it was curated or re-run after tracking ([curated results files](#curated-results-files))
- only `--device` and `--list` apply
- a pre-manifest tracking folder (ROICaT files directly in it) still opens

Viewer:
- up to three sessions side by side
- Sessions tab picks each panel's session and which sessions' traces are drawn
- left / right shift panels one session, `[` / `]` one page
- double-click a footprint to select its cluster in every session
- Clusters table: each cluster's session count, similarity, silhouette, and ROI per visible session

## curated results files

The demixing viewer's Demix (Python: `SingleSessionDemixingVis(..., results_path=...)`, or
`masknmf.demixing.update_signals` then `write_curated`) never changes a results file; it writes a new one beside it:

| file | holds |
|---|---|
| `results.hdf5` | the run: registration, compression and demixing groups |
| `results.<yyyymmddTHHMMSS>.curated.hdf5` | only `DemixingResults`, after the full NMF pass; its `description` attribute says which signals were removed (and by which filter) and how many drawn ROIs were added |
| `results.<yyyymmddTHHMMSS>.curated.labels.hdf5` | the curated file's own labels; the parent's are not copied, curation renumbers the ROIs |

- curating a curated file writes another `results.<later stamp>.curated.hdf5`
- glutamate pipeline: `results.calcium.<stamp>.curated.hdf5` and `results.glutamate.<stamp>.curated.hdf5`
- curated files from before 2026-09-30 are named `<stamp>.curated.hdf5`; rename to `results.<stamp>.curated.hdf5`
- all stamps are ISO 8601 basic (`20261001T183740`), readable with `datetime.fromisoformat`; older `yyyy-mm-dd-HH-MM-SS` stamps still open and sort as older

Which file a command reads:

| RESULTS given as | file used |
|---|---|
| a file path | exactly that file |
| a glob (CLI), or a folder (`masknmf view FOLDER`, classification viewer's Open) | for each results file, its newest curated file when the glob matched one, else the results file; the CLI prints how many it left out |

```bash
masknmf view "sessions/*/*.hdf5" --classify              # newest curated file per run, else results.hdf5
masknmf view "sessions/*/results.hdf5" --classify        # uncurated only
masknmf view sessions/day1/results.20260930T122707.curated.hdf5   # one given version
```

Per command:

- `masknmf view FILE`: a curated file has no registration group, so no shift traces or registered panel; `--raw` still adds the raw panel
- `--classify`, `train-classifier`, `classify`: each file is one session with its own `.labels.hdf5`; a curated file starts unlabeled
- `masknmf track`: clusters index ROIs by position, so a tracking belongs to those exact files. Curate first, then track; re-track any day curated later
- `masknmf view TRACKING_FOLDER [FILES...]`: a file whose ROI count differs from the tracking's is refused
  (`session 1: ... holds 60 ROIs, the tracking 63; the file was curated or re-run after tracking, so track the
  sessions' current files again`)

Python:

| call | does |
|---|---|
| `masknmf.demixing.write_curated(path, results, drop, num_masks, filters)` | writes `<stem>.<stamp>.curated.hdf5` beside `path`, returns its path |
| `masknmf.demixing.results_stem(path)` | the results file a file is or descends from: `results.calcium.<stamp>.curated.hdf5 -> results.calcium` |
| `masknmf.demixing.latest_results(paths)` | the glob rule above: one file per results file, its newest curated one when present |
| `RoicatTrackingResults(..., session_files=...)`, `.session_files = ...`, `from_roicat_dir(..., session_files=...)` | refuse a file whose ROI count differs from the tracking's |

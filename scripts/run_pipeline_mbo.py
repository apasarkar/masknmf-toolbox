import os
from pathlib import Path
import time

import numpy as np
from numpy.typing import ArrayLike
from icecream import ic

import torch
import tifffile
import ffmpeg
from matplotlib import cm
from matplotlib.patches import Rectangle

import masknmf

if "MASKNMF_DEBUG" in os.environ:
    ic.enable()
else:
    ic.disable()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not torch.cuda.is_available():
    print(
        "CUDA is not available. Using CPU instead.\n"
        "If you have a GPU, check your CUDA installation with:\n"
        "  nvcc --version\n"
        "and note the 11x or 12x version (e.g. Build cuda_12.6.r12.6/compiler.34841621_0).\n"
        "To install the correct version of PyTorch with CUDA, go to:\n"
        "  https://pytorch.org/get-started/locally/\n"
        "and select your system configuration to get the appropriate pip or conda command."
    )
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
    print(f"Using CUDA with device: {torch.cuda.get_device_name(0)}")


def norm_minmax(imgs: np.ndarray) -> np.ndarray:
    """
    Normalize a NumPy array to the [0, 1] range.

    Parameters
    ----------
    imgs: np.ndarray) -> np.ndarray: : numpy.ndarray
       The input array to be normalized.

    Returns
    -------
    numpy.ndarray
       The normalized array with values scaled between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([10, 20, 30])
    >>> norm_minmax(arr)
    array([0. , 0.5, 1. ])
    """
    return (imgs - imgs.min()) / (imgs.max() - imgs.min())

def save_mp4(
        fname: str | Path | np.ndarray,
        images,
        framerate=60,
        speedup=1,
        chunk_size=100,
        cmap="gray",
        win=7,
        vcodec="libx264",
        normalize=True,
):
    """
    Save a video from a 3D array or TIFF stack to `.mp4`.

    Parameters
    ----------
    fname : str
        Output video file name.
    images : numpy.ndarray or str
        Input 3D array (T x H x W) or a file path to a TIFF stack.
    framerate : int, optional
        Original framerate of the video, by default 60.
    speedup : int, optional
        Factor to increase the playback speed, by default 1 (no speedup).
    chunk_size : int, optional
        Number of frames to process and write in a single chunk, by default 100.
    cmap : str, optional
        Colormap to apply to the video frames, by default "gray".
        Must be a valid Matplotlib colormap name.
    win : int, optional
        Temporal averaging window size. If `win > 1`, frames are averaged over
        the specified window using convolution. By default, 7.
    vcodec : str, optional
        Video codec to use, by default 'libx264'.
    normalize : bool, optional
        Flag to min-max normalize the video frames, by default True.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist when `images` is provided as a file path.
    ValueError
        If `images` is not a valid 3D NumPy array or a file path to a TIFF stack.

    Notes
    -----
    - The input array `images` must have the shape (T, H, W), where T is the number of frames,
      H is the height, and W is the width.
    - The `win` parameter performs temporal smoothing by averaging over adjacent frames.

    Examples
    --------
    Save a video from a 3D NumPy array with a gray colormap and 2x speedup:

    >>> import numpy as np
    >>> images = np.random.rand(100, 600, 576) * 255
    >>> save_mp4('output.mp4', images, framerate=17, cmap='gray', speedup=2)

    Save a video with temporal averaging applied over a 5-frame window at 4x speed:

    >>> save_mp4('output_smoothed.mp4', images, framerate=30, speedup=4, cmap='gray', win=5)

    Save a video from a TIFF stack:

    >>> save_mp4('output.mp4', 'path/to/stack.tiff', framerate=60, cmap='gray')
    """
    if not isinstance(fname, (str, Path)):
        raise TypeError(f"Expected fname to be str or Path, got {type(fname)}")
    if isinstance(images, (str, Path)):
        print(f"Loading TIFF stack from {images}")
        if Path(images).is_file():
            try:
                images = tifffile.memmap(images)
            except MemoryError:
                images = tifffile.imread(images)
        else:
            raise FileNotFoundError(
                f"Images given as a string or path, but not a valid file: {images}"
            )
    elif not isinstance(images, np.ndarray):
        raise ValueError(
            f"Expected images to be a numpy array or a file path, got {type(images)}"
        )

    T, height, width = images.shape
    colormap = cm.get_cmap(cmap)

    if normalize:
        print("Normalizing mp4 images to [0, 1]")
        images = norm_minmax(images)

    if win and win > 1:
        print(f"Applying temporal averaging with window size {win}")
        kernel = np.ones(win) / win
        images = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=images
        )

    print(f"Saving {T} frames to {fname}")
    output_framerate = int(framerate * speedup)
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            framerate=output_framerate,
        )
        .output(str(fname), pix_fmt="yuv420p", vcodec=vcodec, r=output_framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = images[start:end]
        colored_chunk = (colormap(chunk)[:, :, :, :3] * 255).astype(np.uint8)
        for frame in colored_chunk:
            process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    print(f"Video saved to {fname}")


def plot_pmd_projection(proj, a, savepath=None, fig_label=None, vmax=None, add_scalebar=False, dx=2.0):
    """
    Plot PMD projection with ROIs overlaid similarly to Suite2p plot_projection.

    Parameters
    ----------
    proj : np.ndarray
        Background image (mean or max projection), shape (Ly, Lx).
    a : np.ndarray
        Spatial components, shape (Ly, Lx, n_rois).
    savepath : Path or None
        Where to save the image (optional).
    fig_label : str or None
        Optional label for the figure.
    vmin, vmax : float
        Contrast limits for the projection.
    add_scalebar : bool
        Add a scale bar to the plot.
    dx : float
        Microns per pixel.
    """
    import matplotlib.pyplot as plt
    shape = proj.shape
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')

    vmin = np.nanpercentile(proj, 2)
    vmax = np.nanpercentile(proj, 98) if vmax is None else vmax
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax)
    masks = np.nanmax(a, axis=-1)
    overlay = np.zeros((*shape, 4), dtype=np.float32)
    overlay[..., 1] = 1
    overlay[..., 3] = (masks > 0) * 1.0
    ax.imshow(overlay)

    ax.set_xticks([])
    ax.set_yticks([])
    if fig_label:
        fig_label = fig_label.replace("_", " ").replace("-", " ").replace(".", " ")
        ax.set_ylabel(fig_label, color='white', fontweight='bold', fontsize=12)

    if add_scalebar:
        scale_bar_length = 100 / dx
        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90
        ax.add_patch(Rectangle((scalebar_x, scalebar_y), scale_bar_length, 5,
                               edgecolor='white', facecolor='white'))
        ax.text(scalebar_x + scale_bar_length / 2, scalebar_y - 10,
                "100 μm", color='white', fontsize=10, ha='center', fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, facecolor='black')
        plt.close(fig)
    else:
        plt.show()


def run_plane(data_array: ArrayLike, idx, save_path=None, **kwargs):

    debug = kwargs.get("debug", False)
    if debug:
        ic.enable()

    save_path = Path.home() / ".masknmf" if save_path is None else Path(save_path).expanduser()
    save = True if save_path is not None else False

    save_path.mkdir(exist_ok=True)
    plane_dir = save_path / f"plane{idx}"
    plane_dir.mkdir(exist_ok=True)
    ic(save_path)

    if save:
        ic(
            save_mp4(plane_dir / "reg.mp4", data_array, framerate=ops["fs"], speedup=10, chunk_size=100, cmap="gray", win=3, vcodec="libx264", normalize=True)
        )

    rigid_strategy = masknmf.RigidMotionCorrection(max_shifts=(5, 5))
    pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrection(num_blocks=(32, 32), overlaps=(5, 5), max_rigid_shifts=[5, 5], max_deviation_rigid=[2, 2])

    pwrigid_strategy = masknmf.motion_correction.compute_template(
        data_array, rigid_strategy, num_iterations_piecewise_rigid=1,
        pwrigid_strategy=pwrigid_strategy, device=DEVICE, batch_size=1000
    )
    moco_results = masknmf.RegistrationArray(data_array, pwrigid_strategy, device=DEVICE)
    dense_moco = moco_results[:]
    ic(dense_moco)

    if save:
        save_mp4(plane_dir / "dense_moco.mp4", dense_moco, framerate=ops["fs"], speedup=10, chunk_size=100, cmap="gray", win=3, vcodec="libx264", normalize=True)

    pmd_obj = masknmf.compression.pmd_decomposition(
        dense_moco, [32, 32], dense_moco.shape[0],
        max_components=10, background_rank=10, device=DEVICE,
    )
    np.savez(plane_dir / "pmd_checkpoint1.npz", pmd_obj)
    ic(f"pmd_obj: {pmd_obj.ndim}")

    pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(pmd_obj, device=DEVICE, frame_batch_size=100)
    np.savez(plane_dir / "pmd_checkpoint2.npz", pmd_obj)
    ic(pmd_demixer.state.state_description)

    init_kwargs = {
        'mad_correlation_threshold': 0.85,
        'min_superpixel_size': 5,
        'robust_corr_term': 1,
        'mad_threshold': 1,
        'residual_threshold': 0.3,
        'patch_size': (40, 40),
        'plot_en': True,
        'text': False,
    }
    pmd_demixer.initialize_signals(**init_kwargs, is_custom=False)
    ic(f"Identified {pmd_demixer.results[0].shape[1]} neurons here")
    np.savez(plane_dir / "pmd_checkpoint3.npz", pmd_obj)

    pmd_demixer.lock_results_and_continue()
    ic(pmd_demixer.state.state_description)

    num_iters = 25
    localnmf_params = {
        'maxiter': num_iters,
        'support_threshold': np.linspace(0.95, 0.8, num_iters).tolist(),
        'deletion_threshold': 0.2,
        'ring_model_start_pt': 4,
        'ring_radius': 10,
        'merge_threshold': 0.8,
        'merge_overlap_threshold': 0.8,
        'update_frequency': 4,
        'c_nonneg': True,
        'denoise': False,
        'plot_en': True
    }

    start_time = time.time()
    with torch.no_grad():
        pmd_demixer.demix(**localnmf_params)
    print(f"that took {time.time() - start_time}")
    print(f"after this step {pmd_demixer.results.a.shape[1]} signals identified")

    pmd_demixer.lock_results_and_continue(carry_background=True)

    init_kwargs = {
        'mad_correlation_threshold': 0.9,
        'min_superpixel_size': 5,
        'robust_corr_term': 1,
        'mad_threshold': 0,
        'residual_threshold': 0.3,
        'patch_size': (40, 40),
        'plot_en': True,
        'text': False,
    }

    pmd_demixer.initialize_signals(**init_kwargs, is_custom=False)
    print(f"Identified {pmd_demixer.results[0].shape[1]} neurons here")
    pmd_demixer.lock_results_and_continue(carry_background=True)

    with torch.no_grad():
        pmd_demixer.demix(**localnmf_params)
    print(f"that took {time.time() - start_time}")
    print(f"after this step {pmd_demixer.results.a.shape[1]} signals identified")

    np.savez(plane_dir / "pmd_final.npz", pmd_obj)

    a = pmd_demixer.results.ac_array.export_a()
    c = pmd_demixer.results.ac_array.export_c()
    np.savez(plane_dir / "pmd_final_a.npz", a)
    np.savez(plane_dir / "pmd_final_c.npz", c)
    ic(a.shape, c.shape)
    print("Complete!")


if __name__ == "__main__":
    for i in range(1, 5):
        reg_file = Path(f"D:/demo/suite2p_results/plane{i}/data.bin")
        ops = np.load(Path(f"D:/demo/suite2p_results/plane{i}/ops.npy"), allow_pickle=True).item()
        nt, Lx, Ly = ops["nframes"], ops["Lx"], ops["Ly"]
        data_arr = np.memmap(reg_file, shape=(nt, Lx, Ly), dtype=np.int16)
        run_plane(
            data_array=data_arr,
            idx=i,
            save_path=None  # go to ~/.masknmf
        )
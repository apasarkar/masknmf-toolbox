"""
Pixel mask editor: a hello_imgui window that thresholds the mean and temporal SD images of a movie, cleans the
result, takes brush edits, and saves a boolean (height, width) .npy that is True on the pixels a pipeline uses.

``masknmf mask movie.tif`` opens it; the launcher opens it from the pixel mask row and fills in the saved path.
The mask functions are plain numpy and need no window.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from imgui_bundle import hello_imgui, imgui, immvision

from masknmf.io import load_movie
from masknmf.motion_correction.moco_preprocessing import dilate_saturation_mask
from masknmf.pipelines.subcellular.glutamate_calcium_spines import otsu_threshold


NAMES_IMAGE = ("mean", "sd")
LABELS_IMAGE = {"mean": "Mean", "sd": "Temporal SD"}
COLOR_MASKED = np.array([255.0, 40.0, 40.0], dtype=np.float32)
PERCENTILES_DISPLAY = (1.0, 99.8)
WIDTH_PANEL_EM = 21.0
SIZE_WINDOW = (1150, 780)


@dataclass
class MaskParams:
    """Settings of the automatic mask. Ranges are percentiles of the smoothed image; pixels inside are kept."""

    sigma: float = 1.0
    use_mean: bool = True
    range_mean: tuple[float, float] = (20.0, 100.0)
    use_sd: bool = False
    range_sd: tuple[float, float] = (20.0, 100.0)
    min_region: int = 0
    grow: int = 0


def sample_frames(movie, num_frames: int) -> np.ndarray:
    """
    Evenly spaced frames of a movie.

    Parameters
    ----------
    movie : ArrayLike
        Shape (frames, height, width).
    num_frames : int
        Upper bound on the frames read.

    Returns
    -------
    np.ndarray
        float32, shape (min(num_frames, frames), height, width).
    """
    indices = np.unique(np.linspace(0, movie.shape[0] - 1, min(num_frames, movie.shape[0])).astype(np.int64))
    frames = movie[indices]
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    return np.asarray(frames, dtype=np.float32)


def summary_images(frames: np.ndarray) -> dict[str, np.ndarray]:
    """The mean and temporal SD images of frames, shape (frames, height, width), keyed "mean" and "sd"."""
    return {"mean": frames.mean(axis=0), "sd": frames.std(axis=0)}


def percentile_range_mask(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """True where image lies between its low and high percentiles, inclusive."""
    value_low, value_high = np.percentile(image, [low, high])
    return (image >= value_low) & (image <= value_high)


def otsu_percentile(image: np.ndarray) -> float:
    """The percentile of image at which Otsu's threshold falls."""
    return 100.0 * float(1.0 - otsu_threshold(image).mean())


def drop_small_regions(mask: np.ndarray, min_region: int) -> np.ndarray:
    """mask, boolean of shape (height, width), without its 4-connected regions of fewer than min_region pixels."""
    labels, _ = scipy.ndimage.label(mask)
    sizes = np.bincount(labels.reshape(-1))
    return mask & (sizes[labels] >= min_region)


def clean_mask(keep: np.ndarray, min_region: int, grow: int) -> np.ndarray:
    """
    Remove small regions from a mask, then grow or shrink the masked area.

    Parameters
    ----------
    keep : np.ndarray
        Boolean, shape (height, width). True on kept pixels.
    min_region : int
        Kept and masked regions with fewer pixels than this join their surroundings. 0 leaves all.
    grow : int
        Radius in pixels by which the masked area grows (positive) or shrinks (negative).

    Returns
    -------
    np.ndarray
        Boolean, shape (height, width).
    """
    if min_region > 0:
        keep = drop_small_regions(mask=keep, min_region=min_region)
        keep = ~drop_small_regions(mask=~keep, min_region=min_region)
    if grow > 0:
        keep = ~dilate_saturation_mask(~keep, expansion_value=grow)
    if grow < 0:
        keep = dilate_saturation_mask(keep, expansion_value=-grow)
    return keep


def compute_mask(images: dict[str, np.ndarray], params: MaskParams) -> np.ndarray:
    """
    The automatic mask of a recording.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        The smoothed "mean" and "sd" images, each shape (height, width).
    params : MaskParams

    Returns
    -------
    np.ndarray
        Boolean, shape (height, width). True on kept pixels.
    """
    keep = np.ones(images["mean"].shape, dtype=bool)
    if params.use_mean:
        keep &= percentile_range_mask(image=images["mean"], low=params.range_mean[0], high=params.range_mean[1])
    if params.use_sd:
        keep &= percentile_range_mask(image=images["sd"], low=params.range_sd[0], high=params.range_sd[1])
    return clean_mask(keep=keep, min_region=params.min_region, grow=params.grow)


def paint_disk(target: np.ndarray, row: float, col: float, radius: float, value: int) -> None:
    """Set a disk of target, shape (height, width), to value in place."""
    rows, cols = np.ogrid[: target.shape[0], : target.shape[1]]
    target[(rows - row) ** 2 + (cols - col) ** 2 <= radius**2] = value


def apply_brush(keep: np.ndarray, brush: np.ndarray) -> np.ndarray:
    """keep with the brush edits on top: brush is 1 where painted masked, 2 where painted kept, 0 elsewhere."""
    return (keep & (brush != 1)) | (brush == 2)


def compose_rgb(image: np.ndarray, keep: np.ndarray, opacity: float) -> np.ndarray:
    """
    A greyscale image with the masked pixels tinted red.

    Parameters
    ----------
    image : np.ndarray
        Shape (height, width).
    keep : np.ndarray
        Boolean, shape (height, width). False pixels are tinted.
    opacity : float
        0 shows the image alone, 1 solid red.

    Returns
    -------
    np.ndarray
        uint8, shape (height, width, 3).
    """
    low, high = np.percentile(image, PERCENTILES_DISPLAY)
    grey = 255.0 * np.clip((image - low) / max(high - low, 1e-12), 0.0, 1.0)
    rgb = np.repeat(grey[:, :, None], 3, axis=2).astype(np.float32)
    rgb[~keep] = (1.0 - opacity) * rgb[~keep] + opacity * COLOR_MASKED
    return np.ascontiguousarray(rgb.astype(np.uint8))


class PixelMaskEditor:
    """The editor's state and its window contents."""

    def __init__(self, images: dict[str, np.ndarray], filepath_out: str):
        """
        Parameters
        ----------
        images : dict[str, np.ndarray]
            The "mean" and "sd" images, each shape (height, width).
        filepath_out : str
            Where Apply saves the mask.
        """
        self.images = images
        self.filepath_out = filepath_out
        self.params = MaskParams()
        self.sigma_smoothed = None
        self.images_smoothed = images
        self.brush = np.zeros(images["mean"].shape, dtype=np.uint8)
        self.keep = np.ones(images["mean"].shape, dtype=bool)
        self.shown = "mean"
        self.opacity = 0.55
        self.brush_on = False
        self.brush_radius = 6.0
        self.applied = False
        self.dirty = True
        self.hidden = False
        self.rgb = None
        self.view = immvision.ImageParams()
        self.view.show_options_button = False
        self.view.show_image_info = False
        self.view.show_pixel_info = False
        self.view.show_zoom_buttons = False
        self.view.can_resize = False
        self.view.add_watched_pixel_on_double_click = False
        self.view.draw_values_on_zoomed_pixels = False
        self.view.show_grid = False

    def update(self) -> None:
        """Recompute the mask and its picture after a change."""
        if self.sigma_smoothed != self.params.sigma:
            self.images_smoothed = {
                name: scipy.ndimage.gaussian_filter(image, sigma=self.params.sigma) if self.params.sigma > 0 else image
                for name, image in self.images.items()
            }
            self.sigma_smoothed = self.params.sigma
        self.keep = apply_brush(keep=compute_mask(images=self.images_smoothed, params=self.params), brush=self.brush)
        self.rgb = compose_rgb(image=self.images[self.shown], keep=self.keep, opacity=0.0 if self.hidden else self.opacity)
        self.view.refresh_image = True
        self.dirty = False

    def save(self) -> None:
        """Write the mask to filepath_out."""
        Path(self.filepath_out).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.filepath_out, self.keep)

    def draw_range(self, name: str) -> None:
        """A checkbox, percentile range and Otsu button for one image."""
        used = getattr(self.params, f"use_{name}")
        low, high = getattr(self.params, f"range_{name}")
        changed_used, used = imgui.checkbox(f"{LABELS_IMAGE[name]} percentile range", used)
        imgui.begin_disabled(not used)
        imgui.set_next_item_width(-hello_imgui.em_size(4.5))
        changed_range, low, high = imgui.drag_float_range2(
            f"##range_{name}", low, high, 0.2, 0.0, 100.0, "%.1f", "%.1f", imgui.SliderFlags_.always_clamp
        )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        if imgui.button(f"Otsu##{name}", imgui.ImVec2(-imgui.FLT_MIN, 0)):
            low, high, changed_range = otsu_percentile(image=self.images_smoothed[name]), 100.0, True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Keep the pixels above Otsu's threshold")
        imgui.end_disabled()
        if changed_used or changed_range:
            setattr(self.params, f"use_{name}", used)
            setattr(self.params, f"range_{name}", (low, high))
            self.dirty = True

    def draw_controls(self) -> None:
        """The settings panel."""
        params = self.params
        imgui.push_item_width(-hello_imgui.em_size(7.5))

        imgui.separator_text("Threshold")
        changed, params.sigma = imgui.slider_float("smoothing (px)", params.sigma, 0.0, 10.0, "%.1f")
        self.dirty |= changed
        for name in NAMES_IMAGE:
            self.draw_range(name=name)

        imgui.separator_text("Cleanup")
        changed, params.min_region = imgui.slider_int("min region (px)", params.min_region, 0, 2000)
        self.dirty |= changed
        changed, params.grow = imgui.slider_int("grow mask (px)", params.grow, -10, 10)
        self.dirty |= changed

        imgui.separator_text("Brush")
        _, self.brush_on = imgui.checkbox("Brush (B)", self.brush_on)
        if imgui.is_key_pressed(imgui.Key.b) and not imgui.get_io().want_text_input:
            self.brush_on = not self.brush_on
        _, self.brush_radius = imgui.slider_float("radius (px)", self.brush_radius, 1.0, 50.0, "%.0f")
        imgui.text_disabled("left drag masks, right drag keeps" if self.brush_on else "left drag pans, wheel zooms")
        imgui.begin_disabled(not self.brush.any())
        if imgui.button("Clear brush edits"):
            self.brush[:] = 0
            self.dirty = True
        imgui.end_disabled()

        imgui.separator_text("Display")
        for name in NAMES_IMAGE:
            if imgui.radio_button(LABELS_IMAGE[name], self.shown == name):
                self.shown, self.dirty = name, True
            imgui.same_line()
        imgui.new_line()
        changed, self.opacity = imgui.slider_float("mask opacity", self.opacity, 0.0, 1.0, "%.2f")
        self.dirty |= changed
        hidden = imgui.is_key_down(imgui.Key.space) and not imgui.get_io().want_text_input
        if hidden != self.hidden:
            self.hidden, self.dirty = hidden, True
        imgui.text_disabled("hold Space to hide the mask")
        imgui.pop_item_width()

        imgui.separator_text("Mask")
        num_masked = int(self.keep.size - self.keep.sum())
        imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.35, 1.0), f"{100.0 * num_masked / self.keep.size:.1f}% masked")
        imgui.text_disabled(f"{num_masked:,} of {self.keep.size:,} px; red pixels are left out")

        height_footer = 2 * imgui.get_frame_height_with_spacing() + imgui.get_text_line_height_with_spacing()
        imgui.set_cursor_pos_y(max(imgui.get_cursor_pos_y(), imgui.get_window_height() - height_footer))
        imgui.text_disabled(Path(self.filepath_out).name)
        if imgui.is_item_hovered():
            imgui.set_tooltip(self.filepath_out)
        width_button = (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) / 2
        if imgui.button("Apply", imgui.ImVec2(width_button, 0)):
            self.save()
            self.applied = True
            hello_imgui.get_runner_params().app_shall_exit = True
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(width_button, 0)):
            hello_imgui.get_runner_params().app_shall_exit = True

    def draw_image(self) -> None:
        """The picture, fit to the room left of the panel, with the brush on it."""
        avail = imgui.get_content_region_avail()
        height, width = self.keep.shape
        scale_fit = max(min(avail.x / width, avail.y / height), 0.05)
        self.view.image_display_size = (int(width * scale_fit), int(height * scale_fit))
        self.view.pan_with_mouse = not self.brush_on
        immvision.image("##pixel_mask", self.rgb, self.view)

        mouse = self.view.mouse_info
        if not (self.brush_on and mouse.is_mouse_hovering):
            return
        zoom = float(self.view.zoom_pan_matrix[0][0])
        imgui.get_foreground_draw_list().add_circle(
            imgui.get_mouse_pos(), self.brush_radius * zoom, imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.9))
        )
        value = 1 if imgui.is_mouse_down(imgui.MouseButton_.left) else 2 if imgui.is_mouse_down(imgui.MouseButton_.right) else 0
        if value != 0:
            col, row = mouse.mouse_position
            paint_disk(target=self.brush, row=row, col=col, radius=self.brush_radius, value=value)
            self.dirty = True

    def gui(self) -> None:
        """One frame: the image on the left, the settings on the right."""
        if self.dirty:
            self.update()
        width_panel = hello_imgui.em_size(WIDTH_PANEL_EM)
        imgui.begin_child("image", imgui.ImVec2(-width_panel - imgui.get_style().item_spacing.x, 0))
        self.draw_image()
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("controls", imgui.ImVec2(width_panel, 0))
        self.draw_controls()
        imgui.end_child()


def default_mask_path(filepath_movie: str) -> str:
    """The mask file beside a movie: <movie name>_pixel_mask.npy."""
    path = Path(filepath_movie).expanduser()
    return str(path.with_name(path.stem + "_pixel_mask.npy"))


def edit_pixel_mask(
    filepath_movie: str,
    filepath_out: str | None = None,
    dataset: str | None = None,
    num_frames: int = 400,
) -> str | None:
    """
    Open the editor on a movie and block until it closes.

    Parameters
    ----------
    filepath_movie : str
        Anything ``masknmf.io.load_movie`` reads.
    filepath_out : str, optional
        Where Apply saves the mask. ``None`` saves beside the movie.
    dataset : str, optional
        The dataset holding the movie in an hdf5 file.
    num_frames : int
        Evenly spaced frames the mean and SD images are computed from.

    Returns
    -------
    str | None
        The saved file, or None when the window closed without Apply.
    """
    if filepath_out is None:
        filepath_out = default_mask_path(filepath_movie=filepath_movie)
    frames = sample_frames(movie=load_movie(filepath_movie, dataset), num_frames=num_frames)
    editor = PixelMaskEditor(images=summary_images(frames=frames), filepath_out=filepath_out)
    del frames

    immvision.use_rgb_color_order()
    runner = hello_imgui.RunnerParams()
    runner.app_window_params.window_title = f"Pixel mask - {Path(filepath_movie).name}"
    runner.app_window_params.window_geometry.size = SIZE_WINDOW
    runner.imgui_window_params.tweaked_theme = hello_imgui.ImGuiTweakedTheme(theme=hello_imgui.ImGuiTheme_.darcula_darker)
    runner.ini_disable = True
    runner.callbacks.show_gui = editor.gui
    hello_imgui.run(runner)
    return filepath_out if editor.applied else None

"""
OME-NGFF 0.5 labels store, written beside the file a viewer opened.

``demixing_results.hdf5`` annotates into ``demixing_results.ome.zarr``: a
multiscale summary image at the root with the ROI label image under
``labels/drawn``, so napari and other NGFF readers open it unchanged.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from masknmf.visualization.imgui.labels import UNLABELED
from masknmf.visualization.rois import RoiLabelStore, RoiRecord

__all__ = ["SUFFIX", "LABEL_NAME", "labels_path", "save", "load"]

SUFFIX = ".ome.zarr"
LABEL_NAME = "drawn"
_VERSION = "0.5"
_AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]


def labels_path(path) -> Path:
    """``demixing_results.hdf5`` -> ``demixing_results.ome.zarr``."""
    path = Path(path)
    if path.name.endswith(SUFFIX):
        return path
    return path.with_name(path.with_suffix("").name + SUFFIX)


def _multiscales(name: str) -> list:
    return [
        {
            "name": name,
            "axes": _AXES,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]}
                    ],
                }
            ],
        }
    ]


def _properties(store: RoiLabelStore, label_names: Sequence[str]) -> list:
    names = tuple(label_names)
    out = []
    for i, roi in enumerate(store.rois):
        classed = 0 <= roi.class_index < len(names)
        out.append(
            {
                "label-value": i + 1,
                "class-index": int(roi.class_index),
                "class": names[roi.class_index] if classed else "",
                "area": int(roi.area),
                "note": roi.note,
                "uid": int(roi.uid),
                "source": roi.source,
            }
        )
    return out


def save(
    path,
    store: RoiLabelStore,
    label_names: Sequence[str] = (),
    summary_image: Optional[np.ndarray] = None,
    extra: Optional[dict] = None,
) -> Path:
    """
    Write ``store`` to the labels zarr beside ``path``.

    Args:
        path: the file the viewer opened
        store (RoiLabelStore): ROIs to write
        label_names (Sequence[str]): class names
        summary_image (np.ndarray): ``(Y, X)`` image the labels annotate
        extra (dict): label value -> further ``properties`` keys, e.g. predictions

    Returns:
        Path: the store that was written
    """
    import zarr

    target = labels_path(path)
    image = (
        np.zeros(store.labels.shape, np.float32)
        if summary_image is None
        else np.asarray(summary_image, np.float32)
    )
    properties = _properties(store, label_names)
    for entry in properties:
        entry.update((extra or {}).get(entry["label-value"], {}))

    root = zarr.open_group(str(target), mode="a", zarr_format=3)
    root.create_array("0", data=image, overwrite=True)
    root.attrs["ome"] = {"version": _VERSION, "multiscales": _multiscales("source")}
    root.attrs["masknmf"] = {
        "label_names": list(label_names),
        "next_uid": int(store.next_uid),
    }

    group = root.require_group("labels")
    group.attrs["ome"] = {"version": _VERSION, "labels": [LABEL_NAME]}

    drawn = group.require_group(LABEL_NAME)
    drawn.create_array("0", data=store.labels.astype(np.uint16), overwrite=True)
    drawn.attrs["ome"] = {
        "version": _VERSION,
        "multiscales": _multiscales(LABEL_NAME),
        "image-label": {
            "version": _VERSION,
            "source": {"image": "../../"},
            "colors": [
                {"label-value": i + 1, "rgba": [*store.rgb(i), 255]}
                for i in range(len(store.rois))
            ],
            "properties": properties,
        },
    }
    store.dirty = False
    return target


def load(path, min_pixels: int = 1) -> tuple:
    """
    Read ``(RoiLabelStore, label_names)`` from the labels zarr beside ``path``.

    Returns ``(None, ())`` when there is no store there. A labels zarr written
    by another tool loads too, its ROIs coming back unclassified.
    """
    import zarr

    target = labels_path(path)
    if not (target / "labels" / LABEL_NAME / "zarr.json").is_file():
        return None, ()

    root = zarr.open_group(str(target), mode="r", zarr_format=3)
    drawn = root[f"labels/{LABEL_NAME}"]
    labels = np.asarray(drawn["0"][:], np.uint16)
    image_label = dict(drawn.attrs.get("ome", {})).get("image-label", {})
    props = {
        int(p["label-value"]): p
        for p in image_label.get("properties", ())
        if "label-value" in p
    }
    meta = dict(root.attrs.get("masknmf") or {})

    areas = np.bincount(labels.ravel(), minlength=int(labels.max(initial=0)) + 1)
    store = RoiLabelStore(*labels.shape, min_pixels=min_pixels)
    store.labels = labels
    store.rois = [
        RoiRecord(
            area=int(props.get(v, {}).get("area", areas[v])),
            class_index=int(props.get(v, {}).get("class-index", UNLABELED)),
            note=str(props.get(v, {}).get("note", "")),
            uid=int(props.get(v, {}).get("uid", v)),
            source=str(props.get(v, {}).get("source", "")),
        )
        for v in range(1, len(areas))
    ]
    store.next_uid = int(
        meta.get("next_uid") or max((r.uid for r in store.rois), default=0) + 1
    )
    return store, tuple(meta.get("label_names") or ())

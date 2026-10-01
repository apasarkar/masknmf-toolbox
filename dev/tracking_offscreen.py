"""
Drive the multisession viewer without a window: render frames, select a cluster, press keys and click widgets for
one frame each, and save screenshots beside this file. Underscored names are the viewer's internals.
"""

import os

os.environ["RENDERCANVAS_BACKEND"] = "offscreen"

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from imgui_bundle import imgui

import masknmf
from masknmf.multisession import RoicatTrackingResults
from masknmf.visualization import multisession_vis
from masknmf.visualization.imgui.keybinds import MULTISESSION

FOLDER = "X:/data/yu/multiDay_population_imaging/masknmf-multisession-testdata-20/tracking"
SESSION_FILES = None
OUT = Path(__file__).resolve().parent / "screenshots"

OUT.mkdir(exist_ok=True)
tracking = RoicatTrackingResults.from_roicat_dir(FOLDER, session_files=SESSION_FILES)
viewer = masknmf.MultiSessionDemixingVis(tracking)
viewer.show()
figure = viewer._ndw.figure
figure.canvas.set_logical_size(1600, 1000)
iio.imwrite(OUT / "01_start.png", np.asarray(figure.canvas.draw()))

viewer.set_session(1, 5)
viewer.select_cluster(0)
viewer._set_contours(True)
iio.imwrite(OUT / "02_cluster0_contours.png", np.asarray(figure.canvas.draw()))
print("panels", viewer._panel_session)

# a key for one frame, through the real key handler
original_pressed = multisession_vis.pressed
for key in ("left", "left", "right", "page_next"):
    multisession_vis.pressed = lambda bind, key=key: bind is MULTISESSION[key]
    figure.canvas.draw()
    multisession_vis.pressed = original_pressed
    print(f"{key:>9}", viewer._panel_session)

# the Sessions tab, then session 1's masks checkbox clicked off
original_tab = imgui.begin_tab_item
imgui.begin_tab_item = lambda label, *a, **k: (
    original_tab(label, flags=imgui.TabItemFlags_.set_selected) if label == "Sessions" else original_tab(label, *a, **k)
)
figure.canvas.draw()
imgui.begin_tab_item = original_tab
original_checkbox = imgui.checkbox
imgui.checkbox = lambda label, *a, **k: (True, False) if label == "##masks1" else original_checkbox(label, *a, **k)
figure.canvas.draw()
imgui.checkbox = original_checkbox
print("masks shown", viewer._masks_shown)
iio.imwrite(OUT / "03_sessions_tab.png", np.asarray(figure.canvas.draw()))
print("screenshots in", OUT)

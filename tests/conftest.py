import os

# must be set before anything imports fastplotlib, so the GUI tests get a
# renderer that needs no window
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

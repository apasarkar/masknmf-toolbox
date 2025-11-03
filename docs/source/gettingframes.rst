How to get frames from a video
===============================

You can use square brackets ``[]`` to get frames from your video object, just like a list or a NumPy array.

Here are the most common ways:

.. code-block:: python

    import numpy as np

    # Example: 100 frames, each 480x640 pixels
    video = InMemoryVideo(np.random.rand(100, 480, 640))

Examples
--------

.. code-block:: python

    video[0]
    # → Gets the first frame.
    # → Shape: (480, 640) — one single 2D image.

    video[-1]
    # → Gets the last frame.
    # → Shape: (480, 640).

    video[0:10]
    # → Gets the first 10 frames.
    # → Shape: (10, 480, 640) — 10 images stacked together.

    video[90:100]
    # → Gets the last 10 frames.
    # → Shape: (10, 480, 640).

    video[[0, 10, 20]]
    # → Gets frames 0, 10, and 20.
    # → Shape: (3, 480, 640).

    video[range(0, 100, 5)]
    # → Gets every 5th frame (0, 5, 10, …).
    # → Shape: (20, 480, 640).

    video[:, 100:200, 200:400]
    # → Gets a cropped area from every frame (rows 100–199, columns 200–399).
    # → Shape: (100, 100, 200).

Quick summary
-------------

- Use ``[number]`` to get one frame.
- Use ``[start:end]`` to get a range of frames.
- Use ``[[a, b, c]]`` to get specific frames.
- Use ``[range(...)]`` to get frames in a step pattern.
- Add more dimensions (like ``[frame, y, x]``) to crop inside the frame.
- If you get one frame → result is 2D.
- If you get many frames → result is 3D.

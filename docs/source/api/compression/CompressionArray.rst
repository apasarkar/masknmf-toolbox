.. _api.CompressionArray:

CompressionArray
****************

================
CompressionArray
================
.. currentmodule:: masknmf

Constructor
~~~~~~~~~~~
.. autosummary::
    :toctree: CompressionArray_api

    CompressionArray

Properties
~~~~~~~~~~
.. autosummary::
    :toctree: CompressionArray_api

    CompressionArray.shape
    CompressionArray.dtype
    CompressionArray.ndim
    CompressionArray.compression_rank
    CompressionArray.spatial_compressed
    CompressionArray.temporal_compressed
    CompressionArray.spatial_trend_basis
    CompressionArray.temporal_trend_basis
    CompressionArray.include_trend
    CompressionArray.mean_image
    CompressionArray.noise_variance_image
    CompressionArray.pixel_index_image
    CompressionArray.spatial_compressed_local_projector
    CompressionArray.device
    CompressionArray.rescale
    CompressionArray.flyweight


Methods
~~~~~~~
.. autosummary::
    :toctree: CompressionArray_api

    CompressionArray.calculate_rank_heatmap
    CompressionArray.project_frames
    CompressionArray.export
    CompressionArray.to

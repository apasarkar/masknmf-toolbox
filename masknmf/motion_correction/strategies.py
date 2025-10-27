import math
from abc import ABC, abstractmethod
import torch
from .registration_methods import register_frames_rigid, register_frames_pwrigid
from masknmf.arrays import LazyFrameLoader, ArrayLike
from typing import *
import random
import numpy as np
from tqdm import tqdm

class MotionCorrectionStrategy(ABC):
    """Abstract base class for motion correction strategies."""

    def __init__(self,
                 template: Optional[np.ndarray] = None,
                 batch_size: int = 200):

        self._template = torch.from_numpy(template).float() if template is not None else None
        self._batch_size = batch_size

    @property
    def template(self) -> Optional[torch.tensor]:
        return self._template

    @template.setter
    def template(self, new_template):
        self._template = new_template

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    @abstractmethod
    def pixel_weighting(self):
        pass

    @abstractmethod
    def _correct_singlebatch(self,
                             reference_frames: np.ndarray,
                             target_frames: Optional[np.ndarray] = None,
                             device: str='cpu',
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function should contain the core logic for motion correcting "n" frames, without any batching logic needed.
        Args:
            reference_frames (np.ndarray): Shape (num_frames, height, width).
            target_frames (np.ndarray): Shape (num_frames, height, width).
            device (str): cpu, cuda, etc. which device pytorch computations should occur on
        Returns:
            - motion corrected frames (np.ndarray). Shape (num_frames, height, width)
            - shifts (np.ndarray). Shape depends on what information needs to be conveyed here. Dimension 0 should
                still be number of frames though.
        """
        pass

    def correct(
        self,
        reference_frames: np.ndarray,
        target_frames: Optional[np.ndarray] = None,
        device: str='cpu',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply motion correction. Reference frames is the set of frames that we align to the template (to learn
        shifts, motion displacement fields, etc.) and target_frames is the image stack to which we apply the
        motion stabilization transformation. If target_frames is unspecified, the motion correction is applied to
        reference_frames.

        Two returned values: (1) the motion corrected data (2) the shift (or displacement field information).
        The data format for (2) varies based on method used"""
        if reference_frames.ndim == 2:
            reference_frames = reference_frames[None, ...]
            if target_frames is not None:
                target_frames = target_frames[None, ...]

        num_iters = math.ceil(reference_frames.shape[0] / self.batch_size)
        registered_frame_outputs = []
        frame_shift_outputs = []
        for k in range(num_iters):
            start = k * self.batch_size
            end = min(start + self.batch_size, reference_frames.shape[0])
            reference_subset = reference_frames[start:end]
            if target_frames is not None:
                target_subset = target_frames[start:end]
            else:
                target_subset = None

            if reference_subset.ndim == 2:
                reference_subset = reference_subset[None, :, :]
            if target_subset is not None and target_subset.ndim == 2:
                target_subset = target_subset[None, :, :]
            reg_output = self._correct_singlebatch(reference_subset,
                                                   target_subset,
                                                   device=device,
                                                   **kwargs)
            registered_frame_outputs.append(reg_output[0])
            frame_shift_outputs.append(reg_output[1])
        moco_output = np.concatenate(registered_frame_outputs, axis=0)
        shift_output = np.concatenate(frame_shift_outputs, axis=0)
        return moco_output, shift_output


class RigidMotionCorrector(MotionCorrectionStrategy):
    def __init__(
        self,
        max_shifts: Tuple[int, int],
        template: Optional[np.ndarray] = None,
        pixel_weighting: Optional[np.ndarray] = None,
        batch_size: int = 200
    ):
        super().__init__(template,
                         batch_size=batch_size)
        self._max_shifts = max_shifts
        self._pixel_weighting = torch.from_numpy(pixel_weighting).float() if pixel_weighting is not None else None

    @property
    def max_shifts(self) -> Tuple[int, int]:
        return self._max_shifts

    @property
    def pixel_weighting(self) -> Union[None, torch.tensor]:
        return self._pixel_weighting

    def _correct_singlebatch(self,
                             reference_frames: np.ndarray,
                             target_frames: Optional[np.ndarray] = None,
                             device: str="cpu",
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )
        if target_frames is not None:
            target_frames = target_frames.to(device).float()

        reference_frames = torch.from_numpy(reference_frames).to(device).float()

        outputs = register_frames_rigid(
            reference_frames.to(device).float(),
            self.template.to(device).float(),
            self.max_shifts,
            target_frames=target_frames,
            pixel_weighting=self.pixel_weighting.to(device).float() if self.pixel_weighting is not None else None,
        )
        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

class PiecewiseRigidMotionCorrector(MotionCorrectionStrategy):
    def __init__(
        self,
        num_blocks: Tuple[int, int],
        overlaps: Tuple[int, int],
        max_rigid_shifts: Tuple[int, int],
        max_deviation_rigid: Tuple[int, int],
        template: Optional[np.ndarray] = None,
        pixel_weighting: Optional[np.ndarray] = None,
        batch_size: int = 200
    ):
        super().__init__(template, batch_size)
        self._num_blocks = num_blocks
        self._overlaps = overlaps
        self._max_rigid_shifts = max_rigid_shifts
        self._max_deviation_rigid = max_deviation_rigid
        self._pixel_weighting = torch.from_numpy(pixel_weighting).float() if pixel_weighting is not None else None

    @property
    def num_blocks(self) -> Tuple[int, int]:
        return self._num_blocks

    @property
    def pixel_weighting(self) -> Optional[torch.tensor]:
        return self._pixel_weighting

    @property
    def overlaps(self) -> Tuple[int, int]:
        return self._overlaps

    @property
    def max_rigid_shifts(self) -> Tuple[int, int]:
        return self._max_rigid_shifts

    @property
    def max_deviation_rigid(self) -> Tuple[int, int]:
        return self._max_deviation_rigid

    def _correct_singlebatch(self,
                             reference_frames: np.ndarray,
                             target_frames: Optional[np.ndarray] = None,
                             device: str="cpu",
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )
        if target_frames is not None:
            target_frames = torch.from_numpy(target_frames).to(device).float()

        reference_frames = torch.from_numpy(reference_frames).to(device).float()

        outputs = register_frames_pwrigid(
            reference_frames.to(device),
            self.template.to(device),
            self.num_blocks,
            self.overlaps,
            self.max_rigid_shifts,
            self.max_deviation_rigid,
            target_frames=target_frames,
            pixel_weighting=self.pixel_weighting.to(device) if self.pixel_weighting is not None else None
        )
        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

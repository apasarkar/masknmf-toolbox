from abc import ABC, abstractmethod
import torch
from .registration_methods import register_frames_rigid, register_frames_pwrigid
from typing import *

class MotionCorrectionStrategy(ABC):
    """Abstract base class for motion correction strategies."""

    def __init__(self, template: Optional[torch.tensor] = None):
        self._template = template

    @property
    def template(self) -> Optional[torch.tensor]:
        return self._template

    @template.setter
    def template(self, new_template):
        self._template = new_template

    @abstractmethod
    def correct(self,
                reference_frames: torch.tensor,
                target_frames: Optional[torch.tensor]) -> Tuple[torch.Tensor, torch.tensor]:
        """Apply motion correction. Reference frames is the set of frames that we align to the template (to learn
        shifts, motion displacement fields, etc.) and target_frames is the image stack to which we apply the
        motion stabilization transformation. If target_frames is unspecified, the motion correction is applied to
        reference_frames.

        Two returned values: (1) the motion corrected data (2) the shift (or displacement field information).
        The data format for (2) varies based on method used"""
        pass

class RigidMotionCorrection(MotionCorrectionStrategy):
    def __init__(self, template: torch.tensor,
                 max_shifts: Tuple[int, int]):
        super().__init__(template)
        self._max_shifts = max_shifts

    @property
    def max_shifts(self) -> Tuple[int, int]:
        return self._max_shifts

    def correct(self,
                reference_frames: torch.tensor,
                target_frames: Optional[torch.tensor] = None) -> Tuple[torch.tensor, torch.tensor]:

        if self.template is None:
            raise ValueError("Template is uninitialized. Initialize template, either through constructor or setter first")
        return register_frames_rigid(reference_frames,
                                     self.template,
                                     self.max_shifts,
                                     target_frames = target_frames)

class PiecewiseRigidMotionCorrection(MotionCorrectionStrategy):
    def __init__(self,
                 template: torch.tensor,
                 strides: Tuple[int, int],
                 overlaps: Tuple[int, int],
                 max_rigid_shifts: Tuple[int, int],
                 max_deviation_rigid: int):

        super().__init__(template)
        self._strides = strides
        self._overlaps = overlaps
        self._max_rigid_shifts = max_rigid_shifts
        self._max_deviation_rigid = max_deviation_rigid

    @property
    def strides(self) -> Tuple[int, int]:
        return self._strides

    @property
    def overlaps(self) -> Tuple[int, int]:
        return self._overlaps

    @property
    def max_rigid_shifts(self) -> Tuple[int, int]:
        return self._max_rigid_shifts

    @property
    def max_deviation_rigid(self) -> int:
        return self._max_deviation_rigid

    def correct(self,
                reference_frames: torch.tensor,
                target_frames: Optional[torch.tensor] = None) -> Tuple[torch.tensor, torch.tensor]:

        if self.template is None:
            raise ValueError("Template is uninitialized. Initialize template, either through constructor or setter first")
        return register_frames_pwrigid(reference_frames,
                                       self.template,
                                       self.strides,
                                       self.overlaps,
                                       self.max_rigid_shifts,
                                       self.max_deviation_rigid,
                                       target_frames = target_frames)
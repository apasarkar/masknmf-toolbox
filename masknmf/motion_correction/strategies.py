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
    def correct(
        self,
        reference_frames: torch.tensor,
        target_frames: Optional[torch.tensor] = None,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.tensor]:
        """Apply motion correction. Reference frames is the set of frames that we align to the template (to learn
        shifts, motion displacement fields, etc.) and target_frames is the image stack to which we apply the
        motion stabilization transformation. If target_frames is unspecified, the motion correction is applied to
        reference_frames.

        Two returned values: (1) the motion corrected data (2) the shift (or displacement field information).
        The data format for (2) varies based on method used"""
        pass


class RigidMotionCorrection(MotionCorrectionStrategy):
    def __init__(
        self,
        max_shifts: Tuple[int, int],
        template: Optional[torch.tensor] = None,
        pixel_weighting: Optional[torch.tensor] = None,
    ):
        super().__init__(template)
        self._max_shifts = max_shifts
        self._pixel_weighting = pixel_weighting

    @property
    def max_shifts(self) -> Tuple[int, int]:
        return self._max_shifts

    @property
    def pixel_weighting(self) -> Optional[torch.tensor]:
        return self._pixel_weighting

    def correct(
        self,
        reference_frames: torch.tensor,
        target_frames: Optional[torch.tensor] = None,
        device: str = "cpu",
    ) -> Tuple[torch.tensor, torch.tensor]:
        if self.template is None:
            raise ValueError(
                "Template is uninitialized. Initialize template, either through constructor or setter first"
            )
        if target_frames is not None:
            target_frames = target_frames.to(device)

        if self.pixel_weighting is not None:
            return register_frames_rigid(
                reference_frames.to(device),
                self.template.to(device),
                self.max_shifts,
                target_frames=target_frames,
                pixel_weighting=self.pixel_weighting.to(device),
            )
        else:
            return register_frames_rigid(
                reference_frames.to(device),
                self.template.to(device),
                self.max_shifts,
                target_frames=target_frames,
            )


class PiecewiseRigidMotionCorrection(MotionCorrectionStrategy):
    def __init__(
        self,
        num_blocks: Tuple[int, int],
        overlaps: Tuple[int, int],
        max_rigid_shifts: Tuple[int, int],
        max_deviation_rigid: Tuple[int, int],
        template: Optional[torch.tensor] = None,
        pixel_weighting: Optional[torch.tensor] = None,
    ):
        super().__init__(template)
        self._num_blocks = num_blocks
        self._overlaps = overlaps
        self._max_rigid_shifts = max_rigid_shifts
        self._max_deviation_rigid = max_deviation_rigid
        self._pixel_weighting = pixel_weighting

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

    def correct(
        self,
        reference_frames: torch.tensor,
        target_frames: Optional[torch.tensor] = None,
        device: str = "cpu",
    ) -> Tuple[torch.tensor, torch.tensor]:
        if self.template is None:
            raise ValueError(
                "Template is uninitialized. Initialize template, either through constructor or setter first"
            )
        if target_frames is not None:
            target_frames = target_frames.to(device)


        return register_frames_pwrigid(
                reference_frames.to(device),
                self.template.to(device),
                self.num_blocks,
                self.overlaps,
                self.max_rigid_shifts,
                self.max_deviation_rigid,
                target_frames=target_frames,
                pixel_weighting=self.pixel_weighting.to(device) if self.pixel_weighting is not None else None,
            )

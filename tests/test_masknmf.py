import numpy as np
import torch
import masknmf
import pytest


def test_rigid_motion_gpu():
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3))
    rigid.compute_template(x)
    out = masknmf.RegistrationArray(x, rigid)[:]
    assert out.shape == x.shape

def test_rigid_motion_cpu():
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3))
    rigid.compute_template(x)
    out = masknmf.RegistrationArray(x, rigid)[:]
    assert out.shape == x.shape

import numpy as np
import torch
import masknmf
import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda"))])
def test_rigid_motion(device):
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3), device=device)
    rigid.compute_template(x)
    out = rigid.motion_correct(x)[:]
    assert out.shape == x.shape

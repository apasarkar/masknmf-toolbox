import numpy as np
import torch
import masknmf
import pytest

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_rigid_motion_gpu():
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3))
    rigid.compute_template(x, device=DEVICE)
    out = masknmf.RegistrationArray(x, rigid, device='cuda')[:]
    assert out.shape == x.shape

def test_rigid_motion_cpu():
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3))
    rigid.compute_template(x, device=DEVICE)
    out = masknmf.RegistrationArray(x, rigid, device='cpu')[:]
    assert out.shape == x.shape

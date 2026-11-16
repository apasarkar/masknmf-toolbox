import numpy as np
import masknmf


def test_rigid_motion_registers_an_integer_movie():
    x = (np.random.default_rng(0).random((4, 32, 32)) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3), device="cpu")
    rigid.compute_template(x)
    out = rigid.motion_correct(x)[:]
    assert out.shape == x.shape
    assert np.isfinite(np.asarray(out)).all()

"""A group of signals merges into one: its rank-1 fit, after the signals that were not merged."""

import numpy as np
import pytest
import torch

import masknmf
from masknmf.demixing import update_signals
from masknmf.demixing.signal_demixer import merge_signal_groups
from test_pipelines import cells_movie, close_log

FOV = 64


def test_two_halves_with_one_trace_merge_into_their_union_after_the_other_signals():
    rng = np.random.default_rng(0)
    a = torch.zeros(12, 4)
    a[0:3, 0] = torch.tensor([1.0, 2.0, 3.0])
    a[3:5, 1] = 1.0
    a[5:8, 2] = torch.tensor([4.0, 5.0, 6.0])
    a[8:12, 3] = 1.0
    c = torch.from_numpy(rng.random((50, 4)).astype(np.float32))
    c[:, 2] = c[:, 0]

    a_merged, c_merged = merge_signal_groups(a.to_sparse_csr(), c, [[0, 2]])

    a_merged = a_merged.to_dense()
    assert a_merged.shape == (12, 3) and c_merged.shape == (50, 3)
    torch.testing.assert_close(a_merged[:, :2], a[:, [1, 3]])
    torch.testing.assert_close(c_merged[:, :2], c[:, [1, 3]])
    torch.testing.assert_close(
        a_merged[:, 2:] @ c_merged[:, 2:].T, a[:, [0, 2]] @ c[:, [0, 2]].T, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "merge_indices, drop",
    [([[3]], []), ([[0, 1], [1, 2]], []), ([[0, 1]], [1])],
)
def test_a_group_of_one_or_a_signal_named_twice_is_refused(merge_indices, drop):
    with pytest.raises(ValueError, match="merge"):
        update_signals(None, merge_indices=merge_indices, drop=drop)


def test_a_signal_split_in_two_comes_back_as_one_and_a_dropped_signal_after_it_is_the_one_removed(tmp_path):
    pipeline = masknmf.TwoPhotonCalciumPipeline(output_folder=str(tmp_path / "runs"), device="cpu")
    filepath_results = pipeline.run(cells_movie(), frame_rate=30) / "results.hdf5"
    close_log(pipeline)
    results = masknmf.DemixingResults.from_hdf5(filepath_results)
    a = results.spatial_demixed.to_dense()  ## (pixels, signals)
    c = results.temporal_demixed
    index_split = int((a > 0).sum(dim=0).argmax())
    index_dropped = a.shape[1] - 1 if index_split < a.shape[1] - 1 else 0

    columns = torch.arange(FOV * FOV) % FOV
    is_left = columns <= columns[a[:, index_split] > 0].float().median()
    a_split = torch.cat([a, (a[:, index_split] * ~is_left)[:, None]], dim=1)
    a_split[:, index_split] = a[:, index_split] * is_left
    index_other_half = a_split.shape[1] - 1
    # NOTE: the results have no setter for their signals; the flyweight holds them
    results.flyweight.spatial_demixed = a_split.to_sparse_coo().coalesce()
    results.flyweight.temporal_demixed = torch.cat([c, c[:, [index_split]]], dim=1)

    nmf_config = {k: v for k, v in masknmf.io.read_nmf_config(filepath_results).items() if k != "kind"}
    merged = update_signals(
        results,
        merge_indices=[[index_split, index_other_half]],
        drop=[index_dropped],
        nmf_config=dict(nmf_config, update_frequency=0),
    )

    a_merged = merged.spatial_demixed.to_dense()
    untouched = [k for k in range(a.shape[1]) if k not in (index_split, index_dropped)]
    assert a_merged.shape[1] == a.shape[1] - 1
    # the pass only drops pixels (speckles, weights fit to zero): with update_frequency=0 no support grows
    support_before = torch.cat([a[:, untouched], a[:, [index_split]]], dim=1) > 0
    support_after = a_merged > 0
    assert not (support_after & ~support_before).any()
    assert (support_after.sum(dim=0) >= 0.8 * support_before.sum(dim=0)).all()
    correlation = np.corrcoef(merged.temporal_demixed[:, -1].numpy(), c[:, index_split].numpy())[0, 1]
    assert correlation > 0.99

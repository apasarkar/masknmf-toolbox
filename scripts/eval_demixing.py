import tifffile
import fastplotlib as fpl
from pathlib import Path
from masknmf import load_from_dir

results_path = r"D:\\phasecorr"
contents = list(Path(results_path).iterdir())

raw = tifffile.imread(contents[-1])
patched = tifffile.imread(contents[-2])

raw_results = load_from_dir(contents[0].joinpath("patched_corrected"))
patched_results = load_from_dir(contents[0].joinpath("plane10"))

raw_pmd = raw_results["pmd_demixer"].results
patched_pmd = patched_results["pmd_demixer"].results

def demixing_comparison(
        results1,
        results2,
        device: str,
        v_range: tuple[float, float],
        show_histogram: bool = False,
):
    results1.to(device)
    results2.to(device)

    data = [
        results1.pmd_array,
        results1.colorful_ac_array,
        results2.pmd_array,
        results2.colorful_ac_array,
    ]
    names = [
        "pmd_1",
        "colorful_1",
        "pmd_2",
        "colorful_2",
    ]
    rgb_flags = [False, True, False, True]

    iw = fpl.ImageWidget(
        data=data,
        names=names,
        rgb=rgb_flags,
        histogram_widget=show_histogram,
        graphic_kwargs={"vmin": v_range[0], "vmax": v_range[1]} if v_range else None,
    )

    for i, subplot in enumerate(iw.figure):
        if "colorful" in names[i]:
            ig = subplot["image_widget_managed"]
            ig.vmin = 0
            ig.vmax = 255

    return iw

iw = demixing_comparison(raw_pmd, patched_pmd, "cuda", (-300, 4000))
iw.show()
fpl.loop.run()
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing._adapters._suite2p_adapter import Suite2pResults, suite2p_frame_rate
from masknmf.demixing._adapters._miniali_adapter import MiniALIResults
from masknmf.demixing._adapters._caiman_adapter import CaimanResults, load_caiman_memmap 

ADAPTERS = (Suite2pResults, CaimanResults, MiniALIResults)


def adapter_for(path) -> type[BaseResults] | None:
    from pathlib import Path

    return next((cls for cls in ADAPTERS if cls.matches(Path(path))), None)


def adapter_sources() -> str:
    """
    Lists every adapter's ``source`` as one comma-separated phrase for help text, "or" before the last:
    "a suite2p plane folder or its stat.npy, caiman CNMF.save output, or miniALI's ALI_Result_new.mat".
    """
    sources = [cls.source for cls in ADAPTERS]
    if len(sources) == 1:
        return sources[0]
    return ", ".join(sources[:-1]) + f", or {sources[-1]}"


def adapter_raw_sources() -> str:
    """Lists what --raw must be for each adapter that needs it; empty when none does."""
    return "; ".join(f"{cls.pipeline}: {cls.raw_source}" for cls in ADAPTERS if cls.raw_source)


def adapter_patterns() -> list[str]:
    """Collects the adapters' file dialog patterns, in adapter order."""
    return [pattern for cls in ADAPTERS for pattern in cls.patterns]

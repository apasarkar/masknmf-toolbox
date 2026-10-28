from masknmf.demixing._base_results import BaseResults
from masknmf.demixing._adapters._suite2p_adapter import Suite2pResults, suite2p_frame_rate
from masknmf.demixing._adapters._miniali_adapter import MiniALIResults
from masknmf.demixing._adapters._caiman_adapter import CaimanResults, load_caiman_memmap 

ADAPTERS = (Suite2pResults, CaimanResults)


def adapter_for(path) -> type[BaseResults] | None:
    from pathlib import Path

    return next((cls for cls in ADAPTERS if cls.matches(Path(path))), None)

"""
masknmf: motion correction, compression and demixing of functional imaging data.

The subpackages are imported on first use rather than at import time, so reaching for one
name does not pull in torch, pytorch_lightning and fastplotlib together.
"""

import importlib

from ._version import __version__, version_info


## TODO: Update the arrays import
_SUBPACKAGES = (
    ("masknmf.arrays", None),
    ("masknmf.utils", ("display",)),
    ("masknmf.compression", None),
    ("masknmf.motion_correction", None),
    ("masknmf.demixing", None),
    ("masknmf.visualization", None),
    ("masknmf.diagnostics", None),
    ("masknmf.pipelines", None),
    ("masknmf.pipelines.configs", None),
)


def _exported(module, names):
    """
    The names a subpackage contributes to the masknmf namespace.

    Args:
        module (ModuleType): An imported subpackage
        names (tuple | None): The names to take, or None for everything it exports
    Returns:
        list[str]: The contributed names
    """
    if names is not None:
        return list(names)
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return [name for name in dir(module) if not name.startswith("_")]


def __getattr__(name):
    """
    Resolve a name by importing the subpackage that defines it.

    Submodules resolve first, so masknmf.utils and masknmf.pipelines keep working without
    anything else being imported.

    Args:
        name (str): The attribute being looked up
    Returns:
        Any: The resolved object, also cached in the module namespace
    Raises:
        AttributeError: If no subpackage exports the name
    """
    if name == "__all__":
        names = []
        for path, limit in _SUBPACKAGES:
            names.extend(_exported(module=importlib.import_module(path), names=limit))
        globals()["__all__"] = sorted(set(names))
        return globals()["__all__"]

    try:
        module = importlib.import_module(f"masknmf.{name}")
    except ModuleNotFoundError as error:
        ## a genuine missing dependency inside the subpackage must not look like a typo
        if error.name != f"masknmf.{name}":
            raise
    else:
        globals()[name] = module
        return module

    for path, limit in _SUBPACKAGES:
        module = importlib.import_module(path)
        if name in _exported(module=module, names=limit) and hasattr(module, name):
            globals()[name] = getattr(module, name)
            return globals()[name]

    raise AttributeError(f"module 'masknmf' has no attribute {name!r}")


def __dir__():
    """Every name masknmf exposes, for tab completion and introspection."""
    return sorted(set(list(globals()) + __getattr__("__all__")))

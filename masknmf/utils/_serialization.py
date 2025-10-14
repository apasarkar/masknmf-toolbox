import os
from pathlib import Path
import traceback
from warnings import warn

import numpy as np
import h5py
import torch


def save_dict(d, filename, group, raise_type_fail=True):
    """
    Recursively save a dict to an hdf5 group in a new file.

    Parameters
    ----------
    d: dict
        dict to save as an hdf5 file

    filename : str
        Full path to save the file to. File must not already exist.

    group : str
        group name to save the dict to

    raise_type_fail : bool
        If True: raise an exception if saving a part of the dict fails.
        If False: prints a warning instead and saves the
        object's __str__() return value.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the path specified by the `filename` parameter already exists.

    TypeError
        If a particular entry within the dict cannot be saved to hdf5 AND
        the argument `raise_type_fail` is set to `True`
    """
    if os.path.isfile(filename):
        raise FileExistsError

    with h5py.File(filename, 'w') as h5file:
        _dicts_to_group(h5file, "{}/".format(group), d,
                        raise_meta_fail=raise_type_fail)


def _dicts_to_group(h5file, path, d, raise_meta_fail):
    for key, item in d.items():
        if isinstance(item, torch.Tensor):
            if item.layout is torch.sparse_coo:
                # store this in a new group
                sparse_dict = {
                    "indices": item.indices(),
                    "values": item.values(),
                    "size": np.asarray(item.size()) # size must be specified
                }

                # save to a group with indices and values arrays
                _dicts_to_group(
                    h5file,
                    "{}{}/".format(path, key),
                    sparse_dict,
                    True
                )

                h5file[path + key].attrs["layout"] = "sparse_coo"

            elif item.layout is torch.strided:
                h5file[path + key] = item.cpu().numpy()
                h5file[path + key].attrs["layout"] = "strided"

            else:
                raise TypeError(
                    f"Serialization of Tensor layout: {item.layout} not supported, this should be fixed."
                )

        elif isinstance(item, np.ndarray):

            if item.dtype == np.dtype('O'):
                # see if h5py is ok with it
                try:
                    h5file[path + key] = item
                    # h5file[path + key].attrs['dtype'] = item.dtype.str
                except TypeError:
                    msg = "numpy dtype 'O' for item:\n{}\n" \
                          "not supported by HDF5\n{}" \
                          "".format(item, traceback.format_exc())

                    if raise_meta_fail:
                        raise TypeError(msg)
                    else:
                        h5file[path + key] = str(item)
                        warn("{}, storing whatever str(obj) returns"
                             "".format(msg))

            # numpy array of unicode strings
            elif item.dtype.str.startswith('<U'):
                h5file[path + key] = item.astype(h5py.special_dtype(vlen=str))

                # otherwise h5py doesn't restore the right dtype for str types
                h5file[path + key].attrs['dtype'] = item.dtype.str

            # other types
            else:
                h5file[path + key] = item

        # single pieces of data
        elif isinstance(item, (str, int, np.int8,
                               np.int16, np.int32, np.int64, float,
                               np.float16, np.float32, np.float64)):
            h5file[path + key] = item

        elif isinstance(item, dict):
            _dicts_to_group(
                h5file, "{}{}/".format(path, key), item, raise_meta_fail
            )

        # last resort, try to convert this object
        # to a dict and save its attributes
        elif hasattr(item, '__dict__'):
            _dicts_to_group(
                h5file,
                "{}{}/".format(path, key),
                item.__dict__,
                raise_meta_fail
            )

        else:
            msg = "{} for item: {} not supported by HDF5" \
                  "".format(type(item), item)

            if raise_meta_fail:
                raise TypeError(msg)

            else:
                h5file[path + key] = str(item)
                warn("{}, storing whatever str(obj) returns"
                     "".format(msg))


def load_dict(filename, group):
    """
    Recursively load a dict from an hdf5 group in a file.

    Parameters
    ----------
    filename : str
        full path to the hdf5 file

    group : str
        Name of the group that contains the dict to load

    Returns
    -------
    d : dict
        dict loaded from the specified hdf5 group.
    """

    with h5py.File(filename, 'r') as h5file:
        return _dicts_from_group(h5file, "{}/".format(group))


def _dicts_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if "layout" in item.attrs:
            # it's a torch tensor
            match item.attrs["layout"]:
                case "sparse_coo":
                    # reconstruct sparse_coo
                    indices = item["indices"][()]
                    values = item["values"][()]
                    size = item["size"][()]
                    # needs to a tuple[int] for `torch.sparse_coo_tensor` to be happy
                    size = tuple(size)

                    ans[key] = torch.sparse_coo_tensor(
                        indices=indices,
                        values=values,
                        size=size
                    )

                case "strided":
                    ans[key] = torch.from_numpy(item[()])

                case _:
                    raise TypeError

        elif isinstance(item, h5py.Dataset):
            if item.attrs.__contains__('dtype'):
                ans[key] = item[()].astype(item.attrs['dtype'])
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            ans[key] = _dicts_from_group(h5file, path + key + '/')
    return ans


class Serializer:
    # should specify set of serialized attributes
    _serialized = {}

    def _to_dict(self) -> dict:
        d = {}

        for key in self._serialized:
            val = getattr(self, key)

            # skip None so we don't have to deal with serializing them.
            # HDF5 has no None type, and it can create headaches to properly
            # serialize and de-serialize them. To get around this, any attributes
            # that are None and required for de-serialization should be default None
            # kwargs in the constructor of the class they are serialized from.
            if val is None:
                continue

            if isinstance(val, (tuple, list)):
                val = np.asarray(val)

            d[key] = val

        return d

    def export(self, path: str | Path):
        """
        Export to an HDF5 file.
        Requires ``h5py`` http://docs.h5py.org/

        Parameters
        ----------
        path : str
            Full file path. File must not already exist.

        Raises
        ------
        FileExistsError
            If a file with the same path already exists.
        """

        d = self._to_dict()
        save_dict(d, filename=path, group="data")

    @classmethod
    def from_hdf5(cls, path):
        d = load_dict(path, 'data')
        return cls(**d)

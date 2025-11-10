import os
from pathlib import Path
import traceback
from warnings import warn

import numpy as np
import h5py
import torch
from typing import Optional

def _is_serializable(obj):
    for attr in ["from_hdf5", "export"]:
        if not hasattr(obj, attr):
            raise TypeError(
                f"The object you have passed is not sufficiently array like, "
                f"it lacks the following property or method: {attr}."
            )

def save_dict(d, group: str, filename: Optional[str | Path] = None, hdf5file: Optional[h5py.File] = None,  raise_type_fail=True):
    """
    Recursively save a dict to an hdf5 group in a new file.

    Args:
        d (dict): dict to save as an hdf5 file

        filename (str | Path): Full path to save the file to. File must not already exist.

        group (str): group name to save the dict to

        raise_type_fail (bool): If True: raise an exception if saving a part of the dict fails.
        If False: prints a warning instead and saves the
        object's __str__() return value.

    Returns:
        None

    Raises
        FileExistsError
            If the path specified by the `filename` parameter already exists.

        TypeError
            If a particular entry within the dict cannot be saved to hdf5 AND
            the argument `raise_type_fail` is set to `True`
    """
    if hdf5file is None:
        with h5py.File(filename, 'w') as h5file:
            _dicts_to_group(h5file, f"{group}/", d,
                            raise_meta_fail=raise_type_fail)
    else:
        _dicts_to_group(hdf5file, f"{group}/", d, raise_meta_fail=raise_type_fail)


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

        #Support recursive serialization
        elif _is_serializable(item):
            ## We want to recursively serialize this
            item.export(hdf5file=h5file, group=path, parameter_name=key)
            
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


def load_dict(group: str,
              filename: str | Path | None = None,  
              hdf5file: Optional[h5py.File]=None):
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
    if hdf5file is not None:
        return _dicts_from_group(hdf5file, group)
    with h5py.File(filename, 'r') as hdf5file:
        return _dicts_from_group(hdf5file, "{}/".format(group))


def _dicts_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if key.startswith("masknmf."):
            #This means we are recursively loading an object, so we need to find the module and load it here
            ## Step 1: Take the key string and parse it into a module + a parameter name
            class_path, param_name = key.split(Serializer.ending_str, 1)
            module_path, class_name = class_path.rsplit(".", 1)
            print(f"the module is {module_path} and the class name is {class_name} and the param name is {param_name}")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            subgroup = "{}{}/".format(path, key)
            ans[param_name] = cls.from_hdf5(hdf5file=h5file, group=subgroup)            
            
        elif "layout" in item.attrs:
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
    ending_str = "___"

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

    @classmethod
    def canonical_groupname(cls, parameter_name: str = None):
        final_str = f"{__class__.__module__}.{__class__.__name__}{__class__.ending_str}"
        if parameter_name is not None:
            final_str = final_str + parameter_name
        return final_str

    def export(self, 
               path: str | Path | None = None, 
               hdf5file: Optional[h5py.File] = None, 
               group: str = None,
               parameter_name: str = None):
        """
        Export to an HDF5 file.
        Requires ``h5py`` http://docs.h5py.org/

        Exactly one of path or hdf5file must not be None

        Args:
            path (str): Full file path. File must not already exist.

        Raises
            FileExistsError
                If a file with the same path already exists.
        """
        if path is None and hdf5file is not None:
            d = self._to_dict()
            current_group = self.canonical_groupname(parameter_name=parameter_name)
            if group is not None:
                group = f"{group}{current_group}/"
            else:
                group = current_group
            save_dict(d, group=group, hdf5file=hdf5file)
            
        elif path is not None and hdf5file is None:
            if os.path.isfile(path):
                raise FileExistsError
            d = self._to_dict()
            current_group = self.canonical_groupname(parameter_name=parameter_name)
            save_dict(d, group=current_group, filename=path)
        else:
            ValueError("Specify exactly one of: path and hdf5file")

    @classmethod
    def from_hdf5(cls, 
                  path: str | Path | None = None, 
                  hdf5file: Optional[h5py.File]=None, 
                  group: Optional[str] = None, 
                  **kwargs):
        """Load result from an hdf5 file. Any additional kwargs are passed to the constructor
        
        Args:
            path (Optional[str | Path]): A filepath to a hdf5 from which results are loaded
            hdf5file (Optional[h5py.File]): A hdf5 file reader with "read" permissions to an existing file
            group (Optional[str]): A group path specifying where in the hdf5 file hierarchy the object is located. 
        """
        if group is None:
            group = cls.canonical_groupname()

        if path is None and hdf5file is not None:
            d = load_dict(group, hdf5file=hdf5file)
        elif path is not None and hdf5file is None:
            d = load_dict(group, filename=path)
        else:
            raise ValueError("Specify exactly one of: path and hdf5file")
        
        return cls(**d, **kwargs)


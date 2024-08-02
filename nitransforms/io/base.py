"""Read/write linear transforms."""

from pathlib import Path
import numpy as np
import nibabel as nb
from nibabel import load as loadimg

import h5py

from ..patched import LabeledWrapStruct


def get_xfm_filetype(xfm_file):
    path = Path(xfm_file)
    ext = path.suffix
    if ext == ".gz" and path.name.endswith(".nii.gz"):
        return "nifti"

    file_types = {
        ".nii": "nifti",
        ".h5": "hdf5",
        ".x5": "x5",
        ".txt": "txt",
        ".mat": "txt",
    }
    return file_types.get(ext, "unknown")


def gather_fields(x5=None, hdf5=None, nifti=None, shape=None, affine=None, header=None):
    xfm_fields = {
        "x5": x5,
        "hdf5": hdf5,
        "nifti": nifti,
        "header": header,
        "shape": shape,
        "affine": affine,
    }
    return xfm_fields


def load_nifti(nifti_file):
    nifti_xfm = nb.load(nifti_file)
    xfm_data = nifti_xfm.get_fdata()
    shape = nifti_xfm.shape
    affine = nifti_xfm.affine
    header = getattr(nifti_xfm, "header", None)
    return gather_fields(nifti=xfm_data, shape=shape, affine=affine, header=header)

def load_hdf5(hdf5_file):
    storage = {}

    def get_hdf5_items(name, x5_root):
        if isinstance(x5_root, h5py.Dataset):
            storage[name] = {
                "type": "dataset",
                "attrs": dict(x5_root.attrs),
                "shape": x5_root.shape,
                "data": x5_root[()],
            }
        elif isinstance(x5_root, h5py.Group):
            storage[name] = {
                "type": "group",
                "attrs": dict(x5_root.attrs),
                "members": {},
            }

    with h5py.File(hdf5_file, "r") as f:
        f.visititems(get_hdf5_items)
    if storage:
        hdf5_storage = {"hdf5": storage}
    return hdf5_storage


def load_x5(x5_file):
    load_hdf5(x5_file)


def load_mat(mat_file):
    affine_matrix = np.loadtxt(mat_file)
    affine = nb.affines.from_matvec(affine_matrix[:, :3], affine_matrix[:, 3])
    return gather_fields(affine=affine)


def xfm_loader(xfm_file):
    loaders = {
        "nifti": load_nifti,
        "hdf5": load_hdf5,
        "x5": load_x5,
        "txt": load_mat,
        "mat": load_mat,
    }
    xfm_filetype = get_xfm_filetype(xfm_file)
    loader = loaders.get(xfm_filetype)
    if loader is None:
        raise ValueError(f"Unsupported file type: {xfm_filetype}")
    return loader(xfm_file)

def to_filename(self, filename, fmt="X5"):
    """Store the transform in BIDS-Transforms HDF5 file format (.x5)."""
    with h5py.File(filename, "w") as out_file:
        out_file.attrs["Format"] = "X5"
        out_file.attrs["Version"] = np.uint16(1)
        root = out_file.create_group("/0")
        self._to_hdf5(root)

    return filename

def _to_hdf5(self, x5_root):
    """Serialize this object into the x5 file format."""
    transform_group = x5_root.create_group("TransformGroup")

    """Group '0' containing Affine transform"""
    transform_0 = transform_group.create_group("0")
    transform_0.attrs["Type"] = "Affine"
    transform_0.create_dataset("Transform", data=self._affine)
    transform_0.create_dataset("Inverse", data=np.linalg.inv(self._affine))

    metadata = {"key": "value"}
    transform_0.attrs["Metadata"] = str(metadata)

    """sub-group 'Domain' contained within group '0' """
    domain_group = transform_0.create_group("Domain")
    # domain_group.attrs["Grid"] = self._grid
    # domain_group.create_dataset("Size", data=_as_homogeneous(self._reference.shape))
    # domain_group.create_dataset("Mapping", data=self.mapping)


def _from_x5(self, x5_root):
    variables = {}

    x5_root.visititems(lambda name, x5_root: loader(name, x5_root, variables))

    _transform = variables["TransformGroup/0/Transform"]
    _inverse = variables["TransformGroup/0/Inverse"]
    _size = variables["TransformGroup/0/Domain/Size"]
    _mapping = variables["TransformGroup/0/Domain/Mapping"]

    return _transform, _inverse, _size, _map

class TransformIOError(IOError):
    """General I/O exception while reading/writing transforms."""


class TransformFileError(TransformIOError):
    """Specific I/O exception when a file does not meet the expected format."""


class StringBasedStruct(LabeledWrapStruct):
    """File data structure from text files."""

    def __init__(self, binaryblock=None, endianness=None, check=True):
        """Create a data structure based off of a string."""
        _dtype = getattr(binaryblock, "dtype", None)
        if binaryblock is not None and _dtype == self.dtype:
            self._structarr = binaryblock.copy()
            return
        super().__init__(binaryblock, endianness, check)

    def __array__(self):
        """Return the internal structure array."""
        return self._structarr

    def to_string(self):
        """Convert to a string directly writeable to file."""
        raise NotImplementedError

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        raise NotImplementedError


class LinearTransformStruct(StringBasedStruct):
    """File data structure from linear transforms."""

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        with open(str(filename), "w") as f:
            f.write(self.to_string())

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms internal RAS+ matrix."""
        raise NotImplementedError

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a file given its path."""
        with open(str(filename)) as f:
            string = f.read()
        return cls.from_string(string)

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        return cls.from_string(fileobj.read())

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an affine from a nitransform's RAS+ matrix."""
        raise NotImplementedError


class LinearParameters(LinearTransformStruct):
    """
    A string-based structure for linear transforms.

    Examples
    --------
    >>> lp = LinearParameters()
    >>> np.array_equal(lp.structarr['parameters'], np.eye(4))
    True

    >>> p = np.diag([2., 2., 2., 1.])
    >>> lp = LinearParameters(p)
    >>> np.array_equal(lp.structarr['parameters'], p)
    True

    """

    template_dtype = np.dtype([("parameters", "f8", (4, 4))])
    dtype = template_dtype

    def __init__(self, parameters=None):
        """
        Initialize with default parameters.


        """
        super().__init__()
        self.structarr["parameters"] = np.eye(4)
        if parameters is not None:
            self.structarr["parameters"] = parameters


class BaseLinearTransformList(LinearTransformStruct):
    """A string-based structure for series of linear transforms."""

    template_dtype = np.dtype([("nxforms", "i4")])
    dtype = template_dtype
    _xforms = None
    _inner_type = LinearParameters

    def __init__(self, xforms=None, binaryblock=None, endianness=None, check=True):
        """Initialize with (optionally) a list of transforms."""
        super().__init__(binaryblock, endianness, check)
        self.xforms = [self._inner_type(parameters=mat) for mat in xforms or []]

    @property
    def xforms(self):
        """Get the list of internal transforms."""
        return self._xforms

    @xforms.setter
    def xforms(self, value):
        self._xforms = list(value)

    def __getitem__(self, idx):
        """Allow dictionary access to the transforms."""
        if idx == "xforms":
            return self._xforms
        if idx == "nxforms":
            return len(self._xforms)
        raise KeyError(idx)


class DisplacementsField:
    """A data structure representing displacements fields."""

    @classmethod
    def from_filename(cls, filename):
        """Import a displacements field from a NIfTI file."""
        imgobj = loadimg(str(filename))
        return cls.from_image(imgobj)

    @classmethod
    def from_image(cls, imgobj):
        """Import a displacements field from a nibabel image object."""
        raise NotImplementedError

    @classmethod
    def to_filename(cls, img, filename):
        """Export a displacements field to a NIfTI file."""
        imgobj = cls.to_image(img)
        imgobj.to_filename(filename)

    @classmethod
    def to_image(cls, imgobj):
        """Export a displacements field image from a nitransforms image object."""
        raise NotImplementedError


def _ensure_image(img):
    if isinstance(img, (str, Path)):
        return loadimg(img)
    return img

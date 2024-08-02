"""Read/write x5 transforms."""

from h5py import File as H5File
from nitransforms.io.base import (
    BaseLinearTransformList,
)


class X5Transform:
    """A string-based structure for X5 linear transforms."""

    _transform = None

    def __init__(self, parameters=None, offset=None):
        return

    def __str__(self):
        return

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a X5 file given its path."""
        if str(filename).endswith(".h5"):
            with H5File(str(filename), "r") as hdf:
                return cls.from_h5obj(hdf)

    @classmethod
    def from_h5obj(cls, h5obj):
        """Read the transformations in an X5 file."""
        xfm_list = list(h5obj.keys())

        xfm = xfm_list["Transform"]
        inv = xfm_list["Inverse"]
        coords = xfm_list["Size"]
        map = xfm_list["Mapping"]

        return xfm, inv, coords, map


class X5LinearTransformArray(BaseLinearTransformList):
    """A string-based structure for series of X5 linear transforms."""

    _inner_type = X5Transform

    @property
    def xforms(self):
        """Get the list of internal X5LinearTransforms."""
        return self._xforms

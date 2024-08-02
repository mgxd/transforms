# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Linear transforms."""

import warnings
import numpy as np
from pathlib import Path

from nibabel.affines import from_matvec

from nitransforms.base import (
    ImageGrid,
    TransformBase,
    _as_homogeneous,
    EQUALITY_TOL,
)
from nitransforms.io import get_linear_factory, TransformFileError


class Affine(TransformBase):
    """Represents linear transforms on image data."""

    __slots__ = ("_matrix", "_inverse")

    def __init__(self, matrix=None, reference=None):
        """
        Initialize a linear transform.

        Parameters
        ----------
        matrix : ndarray
            The coordinate transformation matrix **in physical
            coordinates**, mapping coordinates from *reference* space
            into *moving* space.
            This matrix should be provided in homogeneous coordinates.

        Examples
        --------
        >>> xfm = Affine(reference=test_dir / "someones_anatomy.nii.gz")
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        >>> xfm = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 0, 4],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

        """
        super().__init__(reference=reference)
        self._matrix = np.eye(4)
        self._inverse = np.eye(4)

        if matrix is not None:
            matrix = np.array(matrix)
            if matrix.ndim != 2:
                raise TypeError("Affine should be 2D.")
            elif matrix.shape[0] != matrix.shape[1]:
                raise TypeError("Matrix is not square.")
            self._matrix = matrix

            if not np.allclose(self._matrix[3, :], (0, 0, 0, 1)):
                raise ValueError(
                    """The last row of a homogeneus matrix \
should be (0, 0, 0, 1), got %s."""
                    % self._matrix[3, :]
                )

            # Normalize last row
            self._matrix[3, :] = (0, 0, 0, 1)
            self._inverse = np.linalg.inv(self._matrix)

    def __eq__(self, other):
        """
        Overload equals operator.

        Examples
        --------
        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm2 = Affine(xfm1.matrix)
        >>> xfm1 == xfm2
        True

        """
        _eq = np.allclose(self.matrix, other.matrix, rtol=EQUALITY_TOL)
        if _eq and self._reference != other._reference:
            warnings.warn("Affines are equal, but references do not match.")
        return _eq

    def __invert__(self):
        """
        Get the inverse of this transform.

        Example
        -------
        >>> matrix = [[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> Affine(np.linalg.inv(matrix)) == ~Affine(matrix)
        True

        """
        return self.__class__(self._inverse)

    def __len__(self):
        """Enable using len()."""
        return 1 if self._matrix.ndim == 2 else len(self._matrix)

    def __matmul__(self, b):
        """
        Compose two Affines.

        Example
        -------
        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm1 @ ~xfm1 == Affine()
        True

        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm1 @ np.eye(4) == xfm1
        True

        """
        if not isinstance(b, self.__class__):
            _b = self.__class__(b)
        else:
            _b = b

        retval = self.__class__(self.matrix.dot(_b.matrix))
        if _b.reference:
            retval.reference = _b.reference
        return retval

    @property
    def matrix(self):
        """Access the internal representation of this affine."""
        return self._matrix

    @property
    def ndim(self):
        """Access the internal representation of this affine."""
        return self._matrix.ndim + 1

    def map(self, x, inverse=False):
        r"""
        Apply :math:`y = f(x)`.

        Parameters
        ----------
        x : N x D numpy.ndarray
            Input RAS+ coordinates (i.e., physical coordinates).
        inverse : bool
            If ``True``, apply the inverse transform :math:`x = f^{-1}(y)`.

        Returns
        -------
        y : N x D numpy.ndarray
            Transformed (mapped) RAS+ coordinates (i.e., physical coordinates).

        Examples
        --------
        >>> xfm = Affine([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        >>> xfm.map((0,0,0))
        array([[1., 2., 3.]])

        >>> xfm.map((0,0,0), inverse=True)
        array([[-1., -2., -3.]])

        """
        affine = self._matrix
        coords = _as_homogeneous(x, dim=affine.shape[0] - 1).T
        if inverse is True:
            affine = self._inverse
        return affine.dot(coords).T[..., :-1]

    def _to_hdf5(self, x, x5_root):
        """Serialize this object into the x5 file format."""
        transgrp = x5_root.create_group("TransformGroup")
        affine = self._x5group_affine(transgrp)
        coords = self._x5group_domain(x, affine)

        if self._reference:
            self.reference._to_hdf5(x5_root.create_group("Reference"))

        return  # nothing?

    def _x5group_affine(self, TransformGroup):
        """Create group "0" for affine in x5_root/TransformGroup/ according to x5 file format"""
        aff = TransformGroup.create_group("0")
        aff.attrs["Type"] = "affine"  # Should have shape {scalar}
        aff.attrs["Metadata"] = (
            "metadata"  # This is a draft for metadata. Should have shape {scalar}
        )
        aff.create_dataset("Transform", data=[self._matrix])  # Should have shape {3,4}
        aff.create_dataset("Inverse", data=[(~self).matrix])  # Should have shape {4,3}
        return aff

    def _x5group_domain(self, x, transform):
        """Create group "Domain" in x5_root/TransformGroup/0/ according to x5 file format"""
        coords = transform.create_group("Domain")
        coords.attrs["Grid"] = (
            "grid"  # How do I interpet this 'grid'? Should have shape {scalar}
        )
        coords.create_dataset(
            "Size", data=_as_homogeneous(x, dim=self._matrix.shape[0] - 1).T
        )  # Should have shape {3}
        coords.create_dataset(
            "Mapping", data=[self.map(self, x)]
        )  # Should have shape {4,4}
        return coords

    def to_filename(self, filename, fmt="X5", moving=None):
        """Store the transform in the requested output format."""
        writer = get_linear_factory(fmt, is_array=False)

        if fmt.lower() in ("itk", "ants", "elastix"):
            writer.from_ras(self.matrix).to_filename(filename)
        else:
            # Rest of the formats peek into moving and reference image grids
            writer.from_ras(
                self.matrix,
                reference=self.reference,
                moving=ImageGrid(moving) if moving is not None else self.reference,
            ).to_filename(filename)
        return filename

    @classmethod
    def from_filename(cls, filename, fmt=None, reference=None, moving=None):
        """Create an affine from a transform file."""
        fmtlist = [fmt] if fmt is not None else ("itk", "lta", "afni", "fsl")

        if fmt is not None and not Path(filename).exists():
            if fmt != "fsl":
                raise FileNotFoundError(
                    f"[Errno 2] No such file or directory: '{filename}'"
                )
            elif not Path(f"{filename}.000").exists():
                raise FileNotFoundError(
                    f"[Errno 2] No such file or directory: '{filename}[.000]'"
                )

        is_array = cls != Affine
        errors = []
        for potential_fmt in fmtlist:
            if potential_fmt == "itk" and Path(filename).suffix == ".mat":
                is_array = False
                cls = Affine

            try:
                struct = get_linear_factory(
                    potential_fmt, is_array=is_array
                ).from_filename(filename)
            except (TransformFileError, FileNotFoundError) as err:
                errors.append((potential_fmt, err))
                continue

            matrix = struct.to_ras(reference=reference, moving=moving)
            return cls(matrix, reference=reference)

        raise TransformFileError(
            f"Could not open <{filename}> (formats tried: {', '.join(fmtlist)})."
        )

    @classmethod
    def from_matvec(cls, mat=None, vec=None, reference=None):
        """
        Create an affine from a matrix and translation pair.

        Example
        -------
        >>> Affine.from_matvec(vec=(4, 0, 0))  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 4.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        """
        mat = mat if mat is not None else np.eye(3)
        vec = vec if vec is not None else np.zeros((3,))
        return cls(from_matvec(mat, vector=vec), reference=reference)

    def __repr__(self):
        """
        Change representation to the internal matrix.

        Example
        -------
        >>> Affine([
        ...     [1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ... ])  # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 0, 4],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

        """
        return repr(self.matrix)


class LinearTransformsMapping(Affine):
    """Represents a series of linear transforms."""

    def __init__(self, transforms, reference=None):
        """
        Initialize a linear transform mapping.

        Parameters
        ----------
        transforms : :obj:`list`
            The inverse coordinate transformation matrix **in physical
            coordinates**, mapping coordinates from *reference* space
            into *moving* space.
            This matrix should be provided in homogeneous coordinates.

        Examples
        --------
        >>> xfm = LinearTransformsMapping([
        ...     [[1., 0, 0, 1.], [0, 1., 0, 2.], [0, 0, 1., 3.], [0, 0, 0, 1.]],
        ...     [[1., 0, 0, -1.], [0, 1., 0, -2.], [0, 0, 1., -3.], [0, 0, 0, 1.]],
        ... ])
        >>> xfm[0].matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])
        >>> (~xfm)[0].matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[ 1., 0., 0., -1.],
               [ 0., 1., 0., -2.],
               [ 0., 0., 1., -3.],
               [ 0., 0., 0.,  1.]])

        """
        super().__init__(reference=reference)

        self._matrix = np.stack(
            [
                (xfm if isinstance(xfm, Affine) else Affine(xfm)).matrix
                for xfm in transforms
            ],
            axis=0,
        )
        self._inverse = np.linalg.inv(self._matrix)

    def __iter__(self):
        """Enable iterating over the series of transforms."""
        for _m in self.matrix:
            yield Affine(_m, reference=self._reference)

    def __getitem__(self, i):
        """Enable indexed access to the series of matrices."""
        return Affine(self.matrix[i, ...], reference=self._reference)

    def map(self, x, inverse=False):
        r"""
        Apply :math:`y = f(x)`.

        Parameters
        ----------
        x : N x D numpy.ndarray
            Input RAS+ coordinates (i.e., physical coordinates).
        inverse : bool
            If ``True``, apply the inverse transform :math:`x = f^{-1}(y)`.

        Returns
        -------
        y : N x D numpy.ndarray
            Transformed (mapped) RAS+ coordinates (i.e., physical coordinates).

        Examples
        --------
        >>> xfm = LinearTransformsMapping([
        ...     [[1., 0, 0, 1.], [0, 1., 0, 2.], [0, 0, 1., 3.], [0, 0, 0, 1.]],
        ...     [[1., 0, 0, -1.], [0, 1., 0, -2.], [0, 0, 1., -3.], [0, 0, 0, 1.]],
        ... ])
        >>> xfm.matrix
        array([[[ 1.,  0.,  0.,  1.],
                [ 0.,  1.,  0.,  2.],
                [ 0.,  0.,  1.,  3.],
                [ 0.,  0.,  0.,  1.]],
        <BLANKLINE>
               [[ 1.,  0.,  0., -1.],
                [ 0.,  1.,  0., -2.],
                [ 0.,  0.,  1., -3.],
                [ 0.,  0.,  0.,  1.]]])

        >>> y = xfm.map([(0, 0, 0), (-1, -1, -1), (1, 1, 1)])
        >>> y[0, :, :3]
        array([[1., 2., 3.],
               [0., 1., 2.],
               [2., 3., 4.]])

        >>> y = xfm.map([(0, 0, 0), (-1, -1, -1), (1, 1, 1)], inverse=True)
        >>> y[0, :, :3]
        array([[-1., -2., -3.],
               [-2., -3., -4.],
               [ 0., -1., -2.]])


        """
        affine = self.matrix
        coords = _as_homogeneous(x, dim=affine.shape[-1] - 1).T
        if inverse is True:
            affine = self._inverse
        return np.swapaxes(affine.dot(coords), 1, 2)

    def to_filename(self, filename, fmt="X5", moving=None):
        """Store the transform in the requested output format."""
        writer = get_linear_factory(fmt, is_array=True)

        if fmt.lower() in ("itk", "ants", "elastix"):
            writer.from_ras(self.matrix).to_filename(filename)
        else:
            # Rest of the formats peek into moving and reference image grids
            writer.from_ras(
                self.matrix,
                reference=self.reference,
                moving=ImageGrid(moving) if moving is not None else self.reference,
            ).to_filename(filename)
        return filename


def load(filename, fmt=None, reference=None, moving=None):
    """
    Load a linear transform file.

    Examples
    --------
    >>> xfm = load(regress_dir / "affine-LAS.itk.tfm")
    >>> isinstance(xfm, Affine)
    True

    >>> xfm = load(regress_dir / "itktflist.tfm")
    >>> isinstance(xfm, LinearTransformsMapping)
    True

    """
    xfm = LinearTransformsMapping.from_filename(
        filename, fmt=fmt, reference=reference, moving=moving
    )

    if isinstance(xfm, LinearTransformsMapping) and len(xfm) == 1:
        xfm = xfm[0]

    return xfm

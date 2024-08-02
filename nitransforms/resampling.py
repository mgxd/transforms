# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Resampling utilities."""
from pathlib import Path
import numpy as np
from nibabel.loadsave import load as _nbload
from scipy import ndimage as ndi

from nitransforms.base import (
    ImageGrid,
    TransformError,
    SpatialReference,
    _as_homogeneous,
)


def apply(
    transform,
    spatialimage,
    reference=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
    output_dtype=None,
):
    """
    Apply a transformation to an image, resampling on the reference spatial object.

    Parameters
    ----------
    spatialimage : `spatialimage`
        The image object containing the data to be resampled in reference
        space
    reference : spatial object, optional
        The image, surface, or combination thereof containing the coordinates
        of samples that will be sampled.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        Determines how the input image is extended when the resamplings overflows
        a border. Default is 'constant'.
    cval : float, optional
        Constant value for ``mode='constant'``. Default is 0.0.
    prefilter: bool, optional
        Determines if the image's data array is prefiltered with
        a spline filter before interpolation. The default is ``True``,
        which will create a temporary *float64* array of filtered values
        if *order > 1*. If setting this to ``False``, the output will be
        slightly blurred if *order > 1*, unless the input is prefiltered,
        i.e. it is the result of calling the spline filter on the original
        input.
    output_dtype: dtype specifier, optional
        The dtype of the returned array or image, if specified.
        If ``None``, the default behavior is to use the effective dtype of
        the input image. If slope and/or intercept are defined, the effective
        dtype is float64, otherwise it is equivalent to the input image's
        ``get_data_dtype()`` (on-disk type).
        If ``reference`` is defined, then the return value is an image, with
        a data array of the effective dtype but with the on-disk dtype set to
        the input image's on-disk dtype.

    Returns
    -------
    resampled : `spatialimage` or ndarray
        The data imaged after resampling to reference space.

    """
    if reference is not None and isinstance(reference, (str, Path)):
        reference = _nbload(str(reference))

    _ref = (
        transform.reference
        if reference is None
        else SpatialReference.factory(reference)
    )

    if _ref is None:
        raise TransformError("Cannot apply transform without reference")

    if isinstance(spatialimage, (str, Path)):
        spatialimage = _nbload(str(spatialimage))

    data = np.asanyarray(spatialimage.dataobj)

    if data.ndim == 4 and data.shape[-1] != len(transform):
        raise ValueError(
            "The fourth dimension of the data does not match the tranform's shape."
        )

    if data.ndim < transform.ndim:
        data = data[..., np.newaxis]

    # For model-based nonlinear transforms, generate the corresponding dense field
    if hasattr(transform, "to_field") and callable(transform.to_field):
        targets = ImageGrid(spatialimage).index(
            _as_homogeneous(
                transform.to_field(reference=reference).map(_ref.ndcoords.T),
                dim=_ref.ndim,
            )
        )
    else:
        targets = ImageGrid(spatialimage).index(  # data should be an image
            _as_homogeneous(transform.map(_ref.ndcoords.T), dim=_ref.ndim)
        )

    if transform.ndim == 4:
        targets = _as_homogeneous(targets.reshape(-2, targets.shape[0])).T

    resampled = ndi.map_coordinates(
        data,
        targets,
        output=output_dtype,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    if isinstance(_ref, ImageGrid):  # If reference is grid, reshape
        hdr = None
        if _ref.header is not None:
            hdr = _ref.header.copy()
            hdr.set_data_dtype(output_dtype or spatialimage.get_data_dtype())
        moved = spatialimage.__class__(
            resampled.reshape(_ref.shape if data.ndim < 4 else _ref.shape + (-1,)),
            _ref.affine,
            hdr,
        )
        return moved

    return resampled

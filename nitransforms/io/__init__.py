# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Read and write transforms."""
from nitransforms.io import afni, fsl, itk, lta, x5
from nitransforms.io.base import TransformIOError, TransformFileError

__all__ = [
    "afni",
    "fsl",
    "itk",
    "lta",
    "get_linear_factory",
    "TransformFileError",
    "TransformIOError",
]

_IO_TYPES = {
    "itk": (itk, "ITKLinearTransform"),
    "ants": (itk, "ITKLinearTransform"),
    "elastix": (itk, "ITKLinearTransform"),
    "lta": (lta, "FSLinearTransform"),
    "fs": (lta, "FSLinearTransform"),
    "fsl": (fsl, "FSLLinearTransform"),
    "afni": (afni, "AFNILinearTransform"),
    "x5": (x5, "X5Transform"),
}


def get_linear_factory(fmt, is_array=True):
    """Return the type required by a given format."""
    if fmt.lower() not in _IO_TYPES:
        raise TypeError(f"Unsupported transform format <{fmt}>.")

    module, classname = _IO_TYPES[fmt.lower()]
    return getattr(module, f"{classname}{'Array' * is_array}")

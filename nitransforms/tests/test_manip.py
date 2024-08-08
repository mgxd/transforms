# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""
import os
import shutil
from subprocess import check_call
import pytest

import numpy as np
import nibabel as nb
from ..manip import load as _load, TransformChain
from ..linear import Affine
from .test_nonlinear import (
    RMSE_TOL,
    APPLY_NONLINEAR_CMD,
)
from nitransforms.resampling import apply

FMT = {"lta": "fs", "tfm": "itk"}


def test_itk_h5(tmp_path, testdata_path):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    img_fname = testdata_path / "T1w_scanner.nii.gz"
    xfm_fname = (
        testdata_path
        / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )

    xfm = _load(xfm_fname)

    assert len(xfm) == 2

    ref_fname = tmp_path / "reference.nii.gz"
    nb.Nifti1Image(
        np.zeros(xfm.reference.shape, dtype="uint16"), xfm.reference.affine,
    ).to_filename(str(ref_fname))

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD["itk"](
        transform=xfm_fname,
        reference=ref_fname,
        moving=img_fname,
        output="resampled.nii.gz",
        extra="",
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")

    nt_moved = apply(xfm, img_fname, order=0)
    nt_moved.to_filename("nt_resampled.nii.gz")
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < RMSE_TOL

    col_moved = xfm.collapse().apply(img_fname, order=0)
    col_moved.to_filename("nt_collapse_resampled.nii.gz")
    diff = sw_moved.get_fdata() - col_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < RMSE_TOL


@pytest.mark.parametrize("ext0", ["lta", "tfm"])
@pytest.mark.parametrize("ext1", ["lta", "tfm"])
@pytest.mark.parametrize("ext2", ["lta", "tfm"])
def test_collapse_affines(tmp_path, data_path, ext0, ext1, ext2):
    """Check whether affines are correctly collapsed."""
    chain = TransformChain(
        [
            Affine.from_filename(
                data_path
                / "regressions"
                / f"from-fsnative_to-scanner_mode-image.{ext0}",
                fmt=f"{FMT[ext0]}",
            ),
            Affine.from_filename(
                data_path / "regressions" / f"from-scanner_to-bold_mode-image.{ext1}",
                fmt=f"{FMT[ext1]}",
            ),
        ]
    )
    assert np.allclose(
        chain.collapse().matrix,
        Affine.from_filename(
            data_path / "regressions" / f"from-fsnative_to-bold_mode-image.{ext2}",
            fmt=f"{FMT[ext2]}",
        ).matrix,
    )

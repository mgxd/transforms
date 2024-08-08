# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of linear transforms."""
import os
import pytest
import numpy as np
from subprocess import check_call
import shutil
import h5py

import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nitransforms import linear as nitl
from nitransforms import io
from nitransforms.resampling import apply
from .utils import assert_affines_by_filename

RMSE_TOL = 0.1
APPLY_LINEAR_CMD = {
    "fsl": """\
flirt -setbackground 0 -interp nearestneighbour -in {moving} -ref {reference} \
-applyxfm -init {transform} -out {resampled}\
""".format,
    "itk": """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o {resampled} -n NearestNeighbor -t {transform} --float\
""".format,
    "afni": """\
3dAllineate -base {reference} -input {moving} \
-prefix {resampled} -1Dmatrix_apply {transform} -final NN\
""".format,
    "fs": """\
mri_vol2vol --mov {moving} --targ {reference} --lta {transform} \
--o {resampled} --nearest""".format,
}


@pytest.mark.parametrize("matrix", [[0.0], np.ones((3, 3, 3)), np.ones((3, 4)), ])
def test_linear_typeerrors1(matrix):
    """Exercise errors in Affine creation."""
    with pytest.raises(TypeError):
        nitl.Affine(matrix)


def test_linear_typeerrors2(data_path):
    """Exercise errors in Affine creation."""
    with pytest.raises(io.TransformFileError):
        nitl.Affine.from_filename(data_path / "itktflist.tfm", fmt="itk")


def test_linear_filenotfound(data_path):
    """Exercise errors in Affine creation."""
    with pytest.raises(FileNotFoundError):
        nitl.Affine.from_filename("doesnotexist.tfm", fmt="itk")

    with pytest.raises(FileNotFoundError):
        nitl.LinearTransformsMapping.from_filename("doesnotexist.tfm", fmt="itk")

    with pytest.raises(FileNotFoundError):
        nitl.LinearTransformsMapping.from_filename("doesnotexist.mat", fmt="fsl")


def test_linear_valueerror():
    """Exercise errors in Affine creation."""
    with pytest.raises(ValueError):
        nitl.Affine(np.ones((4, 4)))


def test_linear_load_unsupported(data_path):
    """Exercise loading transform without I/O implementation."""
    with pytest.raises(TypeError):
        nitl.load(data_path / "itktflist2.tfm", fmt="X5")


def test_linear_load_mistaken(data_path):
    """Exercise loading transform without I/O implementation."""
    with pytest.raises(io.TransformFileError):
        nitl.load(data_path / "itktflist2.tfm", fmt="afni")


def test_loadsave_itk(tmp_path, data_path, testdata_path):
    """Test idempotency."""
    ref_file = testdata_path / "someones_anatomy.nii.gz"
    xfm = nitl.load(data_path / "itktflist2.tfm", fmt="itk")
    assert isinstance(xfm, nitl.LinearTransformsMapping)
    xfm.reference = ref_file
    xfm.to_filename(tmp_path / "transform-mapping.tfm", fmt="itk")

    assert (data_path / "itktflist2.tfm").read_text() == (
        tmp_path / "transform-mapping.tfm"
    ).read_text()

    single_xfm = nitl.load(data_path / "affine-LAS.itk.tfm", fmt="itk")
    assert isinstance(single_xfm, nitl.Affine)
    assert single_xfm == nitl.Affine.from_filename(
        data_path / "affine-LAS.itk.tfm", fmt="itk"
    )


@pytest.mark.parametrize(
    "image_orientation",
    [
        "RAS",
        "LAS",
        "LPS",
        "oblique",
    ],
)
def test_itkmat_loadsave(tmpdir, data_path, image_orientation):
    tmpdir.chdir()

    io.itk.ITKLinearTransform.from_filename(
        data_path / f"affine-{image_orientation}.itk.tfm"
    ).to_filename(f"affine-{image_orientation}.itk.mat")

    xfm = nitl.load(data_path / f"affine-{image_orientation}.itk.tfm", fmt="itk")
    mat1 = nitl.load(f"affine-{image_orientation}.itk.mat", fmt="itk")

    assert xfm == mat1

    mat2 = nitl.Affine.from_filename(f"affine-{image_orientation}.itk.mat", fmt="itk")

    assert xfm == mat2

    mat3 = nitl.LinearTransformsMapping.from_filename(
        f"affine-{image_orientation}.itk.mat", fmt="itk"
    )

    assert xfm == mat3


@pytest.mark.parametrize("autofmt", (False, True))
@pytest.mark.parametrize("fmt", ["itk", "fsl", "afni", "lta"])
def test_loadsave(tmp_path, data_path, testdata_path, autofmt, fmt):
    """Test idempotency."""
    supplied_fmt = None if autofmt else fmt

    # Load reference transform
    ref_file = testdata_path / "someones_anatomy.nii.gz"
    xfm = nitl.load(data_path / "itktflist2.tfm", fmt="itk")
    xfm.reference = ref_file

    fname = tmp_path / ".".join(("transform-mapping", fmt))
    xfm.to_filename(fname, fmt=fmt)

    if fmt == "fsl":
        # FSL should not read a transform without reference
        with pytest.raises(io.TransformIOError):
            nitl.load(fname, fmt=supplied_fmt)
            nitl.load(fname, fmt=supplied_fmt, moving=ref_file)

        with pytest.warns(UserWarning):
            assert np.allclose(
                xfm.matrix,
                nitl.load(fname, fmt=supplied_fmt, reference=ref_file).matrix,
            )

        assert np.allclose(
            xfm.matrix,
            nitl.load(fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file).matrix,
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)

    xfm.to_filename(fname, fmt=fmt, moving=ref_file)

    if fmt == "fsl":
        assert np.allclose(
            xfm.matrix,
            nitl.load(fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file).matrix,
            rtol=1e-2,  # FSL incurs into large errors due to rounding
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)

    ref_file = testdata_path / "someones_anatomy.nii.gz"
    xfm = nitl.load(data_path / "affine-LAS.itk.tfm", fmt="itk")
    xfm.reference = ref_file
    fname = tmp_path / ".".join(("single-transform", fmt))
    xfm.to_filename(fname, fmt=fmt)
    if fmt == "fsl":
        assert np.allclose(
            xfm.matrix,
            nitl.load(fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file).matrix,
            rtol=1e-2,  # FSL incurs into large errors due to rounding
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)

    xfm.to_filename(fname, fmt=fmt, moving=ref_file)
    if fmt == "fsl":
        assert np.allclose(
            xfm.matrix,
            nitl.load(fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file).matrix,
            rtol=1e-2,  # FSL incurs into large errors due to rounding
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)


@pytest.mark.parametrize("image_orientation", ["RAS", "LAS", "LPS", "oblique"])
@pytest.mark.parametrize("sw_tool", ["itk", "fsl", "afni", "fs"])
def test_linear_save(tmpdir, data_path, get_testdata, image_orientation, sw_tool):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()
    img = get_testdata[image_orientation]
    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    if sw_tool == "fs":
        # Account for the fact that FS defines LTA transforms reversed
        T = np.linalg.inv(T)

    xfm = (
        nitl.Affine(T) if (sw_tool, image_orientation) != ("afni", "oblique") else
        # AFNI is special when moving or reference are oblique - let io do the magic
        nitl.Affine(io.afni.AFNILinearTransform.from_ras(T).to_ras(
            reference=img,
            moving=img,
        ))
    )
    xfm.reference = img

    ext = ""
    if sw_tool == "itk":
        ext = ".tfm"
    elif sw_tool == "fs":
        ext = ".lta"

    xfm_fname1 = f"M.{sw_tool}{ext}"
    xfm.to_filename(xfm_fname1, fmt=sw_tool)

    xfm_fname2 = str(data_path / "affine-%s.%s%s") % (image_orientation, sw_tool, ext)
    assert_affines_by_filename(xfm_fname1, xfm_fname2)


@pytest.mark.parametrize("image_orientation", ["RAS", "LAS", "LPS", 'oblique', ])
@pytest.mark.parametrize("sw_tool", ["itk", "fsl", "afni", "fs"])
def test_apply_linear_transform(tmpdir, get_testdata, get_testmask, image_orientation, sw_tool):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()

    img = get_testdata[image_orientation]
    msk = get_testmask[image_orientation]

    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = nitl.Affine(T)
    xfm.reference = img

    ext = ""
    if sw_tool == "itk":
        ext = ".tfm"
    elif sw_tool == "fs":
        ext = ".lta"

    img.to_filename("img.nii.gz")
    msk.to_filename("mask.nii.gz")

    # Write out transform file (software-dependent)
    xfm_fname = f"M.{sw_tool}{ext}"
    # Change reference dataset for AFNI & oblique
    if (sw_tool, image_orientation) == ("afni", "oblique"):
        io.afni.AFNILinearTransform.from_ras(
            T,
            moving=img,
            reference=img,
        ).to_filename(xfm_fname)
    else:
        xfm.to_filename(xfm_fname, fmt=sw_tool)

    cmd = APPLY_LINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=os.path.abspath("mask.nii.gz"),
        moving=os.path.abspath("mask.nii.gz"),
        resampled=os.path.abspath("resampled_brainmask.nii.gz"),
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    # resample mask
    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved_mask = nb.load("resampled_brainmask.nii.gz")

    nt_moved_mask = apply(xfm, msk, order=0)
    nt_moved_mask.set_data_dtype(msk.get_data_dtype())
    nt_moved_mask.to_filename("ntmask.nii.gz")
    diff = np.asanyarray(sw_moved_mask.dataobj) - np.asanyarray(nt_moved_mask.dataobj)

    assert np.sqrt((diff ** 2).mean()) < RMSE_TOL
    brainmask = np.asanyarray(nt_moved_mask.dataobj, dtype=bool)

    cmd = APPLY_LINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=os.path.abspath("img.nii.gz"),
        moving=os.path.abspath("img.nii.gz"),
        resampled=os.path.abspath("resampled.nii.gz"),
    )

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")
    sw_moved.set_data_dtype(img.get_data_dtype())

    nt_moved = apply(xfm, img, order=0)
    diff = (
        np.asanyarray(sw_moved.dataobj, dtype=sw_moved.get_data_dtype())
        - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    )

    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff[brainmask] ** 2).mean()) < RMSE_TOL

    nt_moved = apply(xfm, "img.nii.gz", order=0)
    diff = (
        np.asanyarray(sw_moved.dataobj, dtype=sw_moved.get_data_dtype())
        - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    )
    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff[brainmask] ** 2).mean()) < RMSE_TOL


def test_Affine_to_x5(tmpdir, testdata_path):
    """Test affine's operations."""
    tmpdir.chdir()
    aff = nitl.Affine()
    with h5py.File("xfm.x5", "w") as f:
        aff._to_hdf5(f.create_group("Affine"))

    aff.reference = testdata_path / "someones_anatomy.nii.gz"
    with h5py.File("withref-xfm.x5", "w") as f:
        aff._to_hdf5(f.create_group("Affine"))


def test_LinearTransformsMapping_apply(tmp_path, data_path, testdata_path):
    """Apply transform mappings."""
    hmc = nitl.load(
        data_path / "hmc-itk.tfm", fmt="itk", reference=testdata_path / "sbref.nii.gz"
    )
    assert isinstance(hmc, nitl.LinearTransformsMapping)

    # Test-case: realign functional data on to sbref
    nii = apply(
        hmc, testdata_path / "func.nii.gz", order=1, reference=testdata_path / "sbref.nii.gz"
    )
    assert nii.dataobj.shape[-1] == len(hmc)

    # Test-case: write out a fieldmap moved with head
    hmcinv = nitl.LinearTransformsMapping(
        np.linalg.inv(hmc.matrix), reference=testdata_path / "func.nii.gz"
    )

    nii = apply(
        hmcinv, testdata_path / "fmap.nii.gz", order=1
    )
    assert nii.dataobj.shape[-1] == len(hmc)

    # Ensure a ValueError is issued when trying to do weird stuff
    hmc = nitl.LinearTransformsMapping(hmc.matrix[:1, ...])
    with pytest.raises(ValueError):
        apply(
            hmc,
            testdata_path / "func.nii.gz",
            order=1,
            reference=testdata_path / "sbref.nii.gz",
        )


def test_mulmat_operator(testdata_path):
    """Check the @ operator."""
    ref = testdata_path / "someones_anatomy.nii.gz"
    mat1 = np.diag([2.0, 2.0, 2.0, 1.0])
    mat2 = from_matvec(np.eye(3), (4, 2, -1))
    aff = nitl.Affine(mat1, reference=ref)

    composed = aff @ nitl.Affine(mat2)
    assert composed.reference is None
    assert composed == nitl.Affine(mat2 @ mat1)

    composed = nitl.Affine(mat2) @ aff
    assert composed.reference == aff.reference
    assert composed == nitl.Affine(mat1 @ mat2, reference=ref)

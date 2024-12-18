import numpy as np
import matplotlib.pyplot as plt
import pytest

import nibabel as nb
from nitransforms.nonlinear import DenseFieldTransform
from nitransforms.vis import PlotDenseField, format_axes


def test_read_path(data_path):
    """Check that filepaths are a supported method for loading
    and reading transforms with PlotDenseField"""
    PlotDenseField(transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")


def test_slice_values():
    """Check that ValueError is issued if negative slices are provided"""
    with pytest.raises(ValueError):
        PlotDenseField(
            transform=np.zeros((10, 10, 10, 3)),
            reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
        ).test_slices(
            xslice=-1,
            yslice=-1,
            zslice=-1,
        )

    "Check that IndexError is issued if provided slices are beyond range of transform dimensions"
    xfm = DenseFieldTransform(
        field=np.zeros((10, 10, 10, 3)),
        reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
    )
    for idx in range(0,3):
        if idx == 0:
            i, j, k = 1, 0, 0
        elif idx == 1:
            i, j, k = 0, 1, 0
        elif idx == 2:
            i, j, k = 0, 0, 1

        with pytest.raises(IndexError):
            PlotDenseField(
                transform=xfm._field,
                reference=xfm._reference,
            ).test_slices(
                xslice=xfm._field.shape[0] + i,
                yslice=xfm._field.shape[1] + j,
                zslice=xfm._field.shape[2] + k,
            )


def test_show_transform(data_path, output_path):
    PlotDenseField(
        transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).show_transform(
        xslice=45,
        yslice=50,
        zslice=55,
    )
    if output_path is not None:
        plt.savefig(output_path / "show_transform.svg", bbox_inches="tight")
    else:
        plt.show()


def test_plot_distortion(data_path, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    PlotDenseField(
        transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_distortion(
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
        show_grid=True,
        show_brain=True,
    )
    if output_path is not None:
        plt.savefig(output_path / "plot_distortion.svg", bbox_inches="tight")
    else:
        plt.show()


def test_empty_quiver():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    PlotDenseField(
        transform=np.zeros((10, 10, 10, 3)),
        reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
    ).plot_quiverdsm(
        axes=axes,
        xslice=5,
        yslice=5,
        zslice=5,
    )


def test_plot_quiverdsm(data_path, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    PlotDenseField(
        transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_quiverdsm(
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
    )

    if output_path is not None:
        plt.savefig(output_path / "plot_quiverdsm.svg", bbox_inches="tight")
    else:
        plt.show()


def test_3dquiver(data_path, output_path):
    with pytest.raises(NotImplementedError):
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        PlotDenseField(
            transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz",
        ).plot_quiverdsm(
            axes=axes,
            xslice=None,
            yslice=None,
            zslice=None,
            three_D=True
        )
        format_axes(axes)

    if output_path is not None:
        plt.savefig(output_path / "plot_3dquiver.svg", bbox_inches="tight")
    else:
        plt.show()


def test_coeffs(data_path, output_path):
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    PlotDenseField(
        transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_coeffs(
        fig=fig,
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
    )

    if output_path is not None:
        plt.savefig(output_path / "plot_coeffs.svg", bbox_inches="tight")
    else:
        plt.show()


def test_plot_jacobian(data_path, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    PlotDenseField(
        transform=data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_jacobian(
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
    )

    if output_path is not None:
        plt.savefig(output_path / "plot_jacobian.svg", bbox_inches="tight")
    else:
        plt.show()

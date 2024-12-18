import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nb

from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from nitransforms.nonlinear import DenseFieldTransform


class PlotDenseField:
    """
    Vizualisation of a transformation file using nitransform's DenseFielTransform module.
    Generates four sorts of plots:
        i) deformed grid superimposed on the normalised deformation field density map\n
        iii) quiver map of the field coloured by its diffusion scalar map\n
        iv) quiver map of the field coloured by the jacobian of the coordinate matrices\n
    for 3 image projections:
        i) axial (fixed z slice)\n
        ii) saggital (fixed y slice)\n
        iii) coronal (fixed x slice)\n
    Outputs the resulting 3 x 3 image grid.

    Parameters
    ----------

    transform: :obj:`str`
        Path from which the trasnformation file should be read.
    is_deltas: :obj:`bool`
        Whether the field is a displacement field or a deformations field. Default = True
    reference : :obj:`ImageGrid`
        Defines the domain of the transform. If not provided, the domain is defined from
        the ``field`` input."""

    __slots__ = ('_transform', '_xfm', '_voxel_size')

    def __init__(self, transform, is_deltas=True, reference=None):
        self._transform = transform
        self._xfm = DenseFieldTransform(
            field=self._transform,
            is_deltas=is_deltas,
            reference=reference
        )
        try:
            """if field provided by path"""
            self._voxel_size = nb.load(transform).header.get_zooms()[:3]
            assert len(self._voxel_size) == 3
        except TypeError:
            """if field provided by numpy array (eg tests)"""
            deltas = []
            for i in range(self._xfm.ndim):
                deltas.append((np.max(self._xfm._field[i]) - np.min(self._xfm._field[i]))
                              / len(self._xfm._field[i]))

            assert np.all(deltas == deltas[0])
            assert len(deltas) == 3
            self._voxel_size = deltas

    def show_transform(
            self,
            xslice,
            yslice,
            zslice,
            scaling=1,
            show_brain=True,
            show_grid=True,
            lw=0.1,
    ):
        """
        Plot output field from DenseFieldTransform class.

        Parameters
        ----------
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        scaling: :obj:`float`
            Fraction by which the quiver plot arrows are to be scaled (default: 1).
        show_brain: :obj:`bool`
            Whether to show the brain image with the deformation grid (default: True).
        show_grid: :obj:`bool`
            Whether to show the deformation grid with the brain deformation (default: True)

        Examples
        --------
        PlotDenseField(
            test_dir / "someones-displacement-field.nii.gz"
        ).show_transform(50, 50, 50)
        plt.show()

        PlotDenseField(
            transform = test_dir / "someones-displacement-field.nii.gz",
            is_deltas = True,
        ).show_transform(
            xslice = 70,
            yslice = 60
            zslice = 90,
            scaling = 3,
            show_brain=False,
            lw = 0.2
            save_to_path = str("test_dir/my_file.jpg"),
        )
        plt.show()
        """
        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)

        fig, axes = format_fig(
            figsize=(9,9),
            gs_rows=3,
            gs_cols=3,
            suptitle="Dense Field Transform \n" + os.path.basename(self._transform),
        )
        fig.subplots_adjust(bottom=0.15)

        projections = ["Axial", "Coronal", "Sagittal"]
        for i, ax in enumerate(axes):
            if i < 3:
                xlabel = None
                ylabel = projections[i]
            else:
                xlabel = ylabel = None
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, labelsize=16)

        self.plot_distortion(
            (axes[2], axes[1], axes[0]),
            xslice,
            yslice,
            zslice,
            show_grid=show_grid,
            show_brain=show_brain,
            lw=lw,
            show_titles=False,
        )
        self.plot_quiverdsm(
            (axes[5], axes[4], axes[3]),
            xslice,
            yslice,
            zslice,
            scaling=scaling,
            show_titles=False,
        )
        self.plot_jacobian(
            (axes[8],axes[7], axes[6]),
            xslice,
            yslice,
            zslice,
            show_titles=False,
        )

        self.sliders(fig, xslice, yslice, zslice)
        # NotImplemented: Interactive slider update here:

    def plot_distortion(
            self,
            axes,
            xslice,
            yslice,
            zslice,
            show_brain=True,
            show_grid=True,
            lw=0.1,
            show_titles=True,
    ):
        """
        Plot the distortion grid.

        Parameters
        ----------
        axis :obj:`tuple`
            Axes on which the grid should be plotted. Requires 3 axes to illustrate
            all projections (eg ax1: Axial, ax2: coronal, ax3: Sagittal)
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        show_brain: :obj:`bool`
            Whether the normalised density map of the distortion should be plotted (Default: True).
        show_grid: :obj:`bool`
            Whether the distorted grid lines should be plotted (Default: True).
        lw: :obj:`float`
            Line width used for gridlines (Default: 0.1).
        show_titles :obj:`bool`
            Show plane names as titles (default: True)

        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            transform="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_distortion(
            axes=[axes[2], axes[1], axes[0]],
            xslice=50,
            yslice=75,
            zslice=90,
            show_brain=True,
            show_grid=True,
            lw=0.2,
        )
        plt.savefig(str("test_dir/deformationgrid.jpg", dpi=300)
        plt.show()
        """
        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)
        planes, titles = self.get_planes(xslice, yslice, zslice)

        for index, plane in enumerate(planes):
            x,y,z,u,v,w = plane
            shape = self._xfm._field.shape[:-1]

            if index == 0:
                dim1, dim2, vec1, vec2 = y, z, v, w
                len1, len2 = shape[1], shape[2]
            elif index == 1:
                dim1, dim2, vec1, vec2 = x, z, u, w
                len1, len2 = shape[0], shape[2]
            else:
                dim1, dim2, vec1, vec2 = x, y, u, v
                len1, len2 = shape[0], shape[1]

            c = np.sqrt(vec1**2 + vec2**2)
            c = c / c.max()

            x_moved = dim1 + vec1
            y_moved = dim2 + vec2

            if show_grid:
                for idx in range(0, len1, 1):
                    axes[index].plot(
                        x_moved[idx * len2:(idx + 1) * len2],
                        y_moved[idx * len2:(idx + 1) * len2],
                        c='k',
                        lw=lw,
                    )
                for idx in range(0, len2, 1):
                    axes[index].plot(
                        x_moved[idx::len2],
                        y_moved[idx::len2],
                        c='k',
                        lw=lw,
                    )

            if show_brain:
                axes[index].scatter(x_moved, y_moved, c=c, cmap='RdPu')

            if show_titles:
                axes[index].set_title(titles[index], fontsize=14, weight='bold')

    def plot_quiverdsm(
            self,
            axes,
            xslice,
            yslice,
            zslice,
            scaling=1,
            three_D=False,
            show_titles=True,
    ):
        """
        Plot the Diffusion Scalar Map (dsm) as a quiver plot.

        Parameters
        ----------
        axis :obj:`tuple`
            Axes on which the quiver should be plotted. Requires 3 axes to illustrate
            the dsm mapped as a quiver plot for each projection.
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal projection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal projection of the transform.
        scaling: :obj:`float`
            Fraction by which the quiver plot arrows are to be scaled (default: 1).
        three_D: :obj:`bool`
            Whether the quiver plot is to be projected onto a 3D axis (default: False)
        show_titles :obj:`bool`
            Show plane names as titles (default: True)

        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            transform="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_quiverdsm(
            axes=[axes[2], axes[1], axes[0]],
            xslice=50,
            yslice=75,
            zslice=90,
            scaling=2,
        )
        plt.savefig(str("test_dir/quiverdsm.jpg", dpi=300)
        plt.show()

        #Example 2: 3D quiver
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        PlotDenseField(transform, is_deltas=True).plot_quiverdsm(
            ax,
            xslice=None,
            yslice=None,
            zslice=None,
            scaling=10,
            three_D=True,
        )
        plt.show()
        """
        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)
        planes, titles = self.get_planes(xslice, yslice, zslice)

        if three_D is not False:
            raise NotImplementedError("3d Quiver plot not finalised.")

            # finalise 3d quiver below:
            for i, j in enumerate(planes):
                x, y, z, u, v, w = j

            magnitude = np.sqrt(u**2 + v**2 + w**2)
            clr3d = plt.cm.viridis(magnitude / magnitude.max())
            xyz = axes.quiver(x, y, z, u, v, w, colors=clr3d, length=1 / scaling)
            plt.colorbar(xyz)
        else:
            for index, plane in enumerate(planes):
                x, y, z, u, v, w = plane
                c_reds, c_greens, c_blues, zeros = [], [], [], []

                # Optimise here, matrix operations
                for idx, (i, j, k, l, m, n) in enumerate(zip(x, y, z, u, v, w)):
                    if np.abs(u[idx]) > [np.abs(v[idx]) and np.abs(w[idx])]:
                        c_reds.append((i, j, k, l, m, n, np.abs(u[idx])))
                    elif np.abs(v[idx]) > [np.abs(u[idx]) and np.abs(w[idx])]:
                        c_greens.append((i, j, k, l, m, n, np.abs(v[idx])))
                    elif np.abs(w[idx]) > [np.abs(u[idx]) and np.abs(v[idx])]:
                        c_blues.append((i, j, k, l, m, n, np.abs(w[idx])))
                    else:
                        zeros.append(0)

                '''Check if shape of c_arrays is (0,) ie transform is independent of some dims'''
                if np.shape(c_reds) == (0,):
                    c_reds = np.zeros((1, 7))
                if np.shape(c_greens) == (0,):
                    c_greens = np.zeros((1, 7))
                if np.shape(c_blues) == (0,):
                    c_blues = np.zeros((1, 7))
                elif (
                    np.shape(c_reds) != (0,)
                    and np.shape(c_greens) != (0,)
                    and np.shape(c_blues) != (0,)
                ):
                    assert len(np.concatenate((c_reds, c_greens, c_blues))) == len(x) - len(zeros)

                c_reds = np.asanyarray(c_reds)
                c_greens = np.asanyarray(c_greens)
                c_blues = np.asanyarray(c_blues)

                if index == 0:
                    dim1, dim2, vec1, vec2 = 1, 2, 4, 5
                elif index == 1:
                    dim1, dim2, vec1, vec2 = 0, 2, 3, 5
                elif index == 2:
                    dim1, dim2, vec1, vec2 = 0, 1, 3, 4

                axes[index].quiver(
                    c_reds[:, dim1],
                    c_reds[:, dim2],
                    c_reds[:, vec1],
                    c_reds[:, vec2],
                    c_reds[:, -1],
                    cmap='Reds',
                )
                axes[index].quiver(
                    c_greens[:, dim1],
                    c_greens[:, dim2],
                    c_greens[:, vec1],
                    c_greens[:, vec2],
                    c_greens[:, -1],
                    cmap='Greens',
                )
                axes[index].quiver(
                    c_blues[:, dim1],
                    c_blues[:, dim2],
                    c_blues[:, vec1],
                    c_blues[:, vec2],
                    c_blues[:, -1],
                    cmap='Blues',
                )

                if show_titles:
                    axes[index].set_title(titles[index], fontsize=14, weight='bold')

    def plot_coeffs(self, fig, axes, xslice, yslice, zslice, s=0.1, show_titles=True):
        """
        Plot linear, planar and spherical coefficients.
        Parameters
        ----------
        fig :obj:`figure`
            Figure to use for mapping the coefficients.
        axis :obj:`tuple`
            Axes on which the quiver should be plotted. Requires 3 axes to illustrate
            the dsm mapped as a quiver plot for each projection.
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal projection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal projection of the transform.
        s: :obj:`float`
            Size of scatter points (default: 0.1).
        show_titles :obj:`bool`
            Show plane names as titles (default: True)

        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            transform="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_coeffs(
            fig=fig
            axes=axes,
            xslice=50,
            yslice=75,
            zslice=90,
        )
        plt.show()
        """
        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)
        planes, titles = self.get_planes(xslice, yslice, zslice)

        for index, plane in enumerate(planes):
            x, y, z, u, v, w = plane

            if index == 0:
                dim1, dim2 = y, z
            elif index == 1:
                dim1, dim2 = x, z
            else:
                dim1, dim2 = x, y

            cl_arr, cp_arr, cs_arr = [], [], []

            for idx, (i, j, k) in enumerate(zip(u, v, w)):
                i, j, k = abs(i), abs(j), abs(k)
                L1, L2, L3 = sorted([i, j, k], reverse=True)
                asum = np.sum([i, j, k])

                cl = (L1 - L2) / asum
                cl_arr.append(cl) if cl != np.nan else cl.append(0)

                cp = 2 * (L2 - L3) / asum
                cp_arr.append(cp) if cp != np.nan else cp.append(0)

                cs = 3 * L3 / asum
                cs_arr.append(cs) if cs != np.nan else cs.append(0)

            a = axes[0, index].scatter(dim1, dim2, c=cl_arr, cmap='Reds', s=s)
            b = axes[1, index].scatter(dim1, dim2, c=cp_arr, cmap='Greens', s=s)
            c = axes[2, index].scatter(dim1, dim2, c=cs_arr, cmap='Blues', s=s)

            if show_titles:
                axes[0, index].set_title(titles[index], fontsize=14, weight='bold')

        cb = fig.colorbar(a, ax=axes[0,:], location='right')
        cb.set_label(label=r"$c_l$",weight='bold', fontsize=14)

        cb = fig.colorbar(b, ax=axes[1,:], location='right')
        cb.set_label(label=r"$c_p$",weight='bold', fontsize=14)

        cb = fig.colorbar(c, ax=axes[2,:], location='right')
        cb.set_label(label=r"$c_s$",weight='bold', fontsize=14)

    def plot_jacobian(self, axes, xslice, yslice, zslice, show_titles=True):
        """
        Map the divergence of the transformation field using a quiver plot.

        Parameters
        ----------
        axis :obj:`tuple`
            Axes on which the quiver should be plotted. Requires 3 axes to illustrate
            each projection (eg ax1: Axial, ax2: coronal, ax3: Sagittal)
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal projection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal projection of the transform.
        show_titles :obj:`bool`
            Show plane names as titles (default: True)

        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            transform="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_jacobian(
            axes=[axes[2], axes[1], axes[0]],
            xslice=50,
            yslice=75,
            zslice=90,
        )
        plt.savefig(str("test_dir/jacobians.jpg", dpi=300)
        plt.show()
        """
        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)
        planes, titles = self.get_planes(xslice, yslice, zslice)

        jacobians = np.zeros((3), dtype=np.ndarray)

        """iterating through the three chosen planes to calculate corresponding coordinates"""
        jac = self.get_jacobian().reshape(self._xfm._field[..., -1].shape)
        for idx, slicer in enumerate((
            (xslice, slice(None), slice(None), None),
            (slice(None), yslice, slice(None), None),
            (slice(None), slice(None), zslice, None),
        )):
            jacobians[idx] = jac[slicer].flatten()

        for index, (ax, plane) in enumerate(zip(axes, planes)):
            x, y, z, _, _, _ = plane

            if index == 0:
                dim1, dim2 = y, z
            elif index == 1:
                dim1, dim2 = x, z
            else:
                dim1, dim2 = x, y

            c = jacobians[index]
            plot = ax.scatter(dim1, dim2, c=c, norm=mpl.colors.CenteredNorm(), cmap='seismic')

            if show_titles:
                ax.set_title(titles[index], fontsize=14, weight='bold')
                plt.colorbar(plot, location='bottom', orientation='horizontal', label=str(r'$J$'))

    def test_slices(self, xslice, yslice, zslice):
        """Ensure slices are positive and within range of image dimensions"""
        xfm = self._xfm._field

        try:
            if xslice < 0 or yslice < 0 or zslice < 0:
                raise ValueError("Slice values must be positive integers")

            if int(xslice) > xfm.shape[0]:
                raise IndexError(f"x-slice {xslice} out of range for transform object "
                                 f"with x-dimension of length {xfm.shape[0]}")
            if int(yslice) > xfm.shape[1]:
                raise IndexError(f"y-slice {yslice} out of range for transform object "
                                 f"with y-dimension of length {xfm.shape[1]}")
            if int(zslice) > xfm.shape[2]:
                raise IndexError(f"z-slice {zslice} out of range for transform object "
                                 f"with z-dimension of length {xfm.shape[2]}")

            return (int(xslice), int(yslice), int(zslice))

        except TypeError as e:
            """exception for case of 3d quiver plot"""
            assert str(e) == "'<' not supported between instances of 'NoneType' and 'int'"

            return (xslice, yslice, zslice)

    def get_coords(self):
        """Calculate vector components of the field using the reference coordinates"""
        x = self._xfm.reference.ndcoords[0].reshape(np.shape(self._xfm._field[...,-1]))
        y = self._xfm.reference.ndcoords[1].reshape(np.shape(self._xfm._field[...,-1]))
        z = self._xfm.reference.ndcoords[2].reshape(np.shape(self._xfm._field[...,-1]))
        u = self._xfm._field[..., 0] - x
        v = self._xfm._field[..., 1] - y
        w = self._xfm._field[..., 2] - z
        return x, y, z, u, v, w

    def get_jacobian(self):
        """Calculate the Jacobian matrix of the field"""
        x, y, z, u, v, w = self.get_coords()
        voxx, voxy, voxz = self._voxel_size

        shape = self._xfm._field[..., -1].shape
        zeros = np.zeros(shape)
        jacobians = zeros.flatten()

        dxdx = (np.diff(u, axis=0) / voxx)
        dydx = (np.diff(v, axis=0) / voxx)
        dzdx = (np.diff(w, axis=0) / voxx)

        dxdy = (np.diff(u, axis=1) / voxy)
        dydy = (np.diff(v, axis=1) / voxy)
        dzdy = (np.diff(w, axis=1) / voxy)

        dxdz = (np.diff(u, axis=2) / voxz)
        dydz = (np.diff(v, axis=2) / voxz)
        dzdz = (np.diff(w, axis=2) / voxz)

        partials = [dxdx, dydx, dzdx, dxdy, dydy, dzdy, dxdz, dydz, dzdz]

        for idx, j in enumerate(partials):
            if idx < 3:
                dim = zeros[-1,:,:][None,:,:]
                ax = 0
            elif idx >= 3 and idx < 6:
                dim = zeros[:,-1,:][:,None,:]
                ax = 1
            elif idx >= 6:
                dim = zeros[:,:,-1][:,:,None]
                ax = 2

            partials[idx] = np.append(j, dim, axis=ax).flatten()

        dxdx, dydx, dzdx, dxdy, dydy, dzdy, dxdz, dydz, dzdz = partials

        for idx, k in enumerate(jacobians):
            jacobians[idx] = np.linalg.det(
                np.array(
                    [
                        [dxdx[idx], dxdy[idx], dxdz[idx]],
                        [dydx[idx], dydy[idx], dydz[idx]],
                        [dzdx[idx], dzdy[idx], dzdz[idx]]
                    ]
                )
            )
        return jacobians

    def get_planes(self, xslice, yslice, zslice):
        """Define slice selection for visualisation"""
        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)
        titles = ["Sagittal", "Coronal", "Axial"]
        planes = [0] * 3
        slices = [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]

        for idx, s in enumerate(slices):
            x, y, z, u, v, w = self.get_coords()

            """indexing for plane selection [x: sagittal, y: coronal, z: axial, dimension]"""
            s = [xslice, slice(None), slice(None), None] if idx == 0 else s
            s = [slice(None), yslice, slice(None), None] if idx == 1 else s
            s = [slice(None), slice(None), zslice, None] if idx == 2 else s
            # For 3d quiver:
            if xslice == yslice == zslice is None:
                s = [slice(None), slice(None), slice(None), None]

            """computing coordinates within each plane"""
            x = x[s[0], s[1], s[2]]
            y = y[s[0], s[1], s[2]]
            z = z[s[0], s[1], s[2]]
            u = self._xfm._field[s[0], s[1], s[2], 0] - x
            v = self._xfm._field[s[0], s[1], s[2], 1] - y
            w = self._xfm._field[s[0], s[1], s[2], 2] - z

            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            u = u.flatten()
            v = v.flatten()
            w = w.flatten()

            """check indexing has retrieved correct dimensions"""
            if idx == 0 and xslice is not None:
                assert x.shape == u.shape == np.shape(self._xfm._field[-1,...,-1].flatten())
            elif idx == 1 and yslice is not None:
                assert y.shape == v.shape == np.shape(self._xfm._field[:,-1,:,-1].flatten())
            elif idx == 2 and zslice is not None:
                assert z.shape == w.shape == np.shape(self._xfm._field[...,-1,-1].flatten())

            """store 3 slices of datapoints, with overall shape [3 x [6 x [data]]]"""
            planes[idx] = [x, y, z, u, v, w]
        return planes, titles

    def sliders(self, fig, xslice, yslice, zslice):
        # This successfully generates a slider, but it cannot be used.
        # Currently, slider only acts as a label to show slice values.
        # raise NotImplementedError("Slider implementation not finalised.
        # Static slider can be generated but is not interactive")

        xslice, yslice, zslice = self.test_slices(xslice, yslice, zslice)
        slices = [
            [zslice, len(self._xfm._field[0][0]), "zslice"],
            [yslice, len(self._xfm._field[0]), "yslice"],
            [xslice, len(self._xfm._field), "xslice"],
        ]
        axes = [
            [1 / 7, 0.1, 1 / 7, 0.025],
            [3 / 7, 0.1, 1 / 7, 0.025],
            [5 / 7, 0.1, 1 / 7, 0.025],
        ]
        sliders = []

        for index, slider_axis in enumerate(axes):
            slice_dim = slices[index][0]
            sax = fig.add_axes(slider_axis)
            slider = Slider(
                ax=sax,
                valmin=0,
                valmax=slices[index][1],
                valinit=slice_dim,
                valstep=1,
                valfmt='%d',
                label=slices[index][2],
                orientation="horizontal"
            )
            sliders.append(slider)

            assert sliders[index].val == slices[index][0]

        return sliders

    def update_sliders(self, slider):
        raise NotImplementedError("Interactive sliders not implemented.")

        new_slider = slider.val
        return new_slider


def format_fig(figsize, gs_rows, gs_cols, **kwargs):
    params = {
        'gs_wspace': 0,
        'gs_hspace': 1 / 8,
        'suptitle': None,
    }
    params.update(kwargs)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        params['suptitle'],
        fontsize='20',
        weight='bold')

    gs = GridSpec(
        gs_rows,
        gs_cols,
        figure=fig,
        wspace=params['gs_wspace'],
        hspace=params['gs_hspace']
    )

    axes = []
    for j in range(0, gs_cols):
        for i in range(0, gs_rows):
            axes.append(fig.add_subplot(gs[i,j]))
    return fig, axes


def format_axes(axis, **kwargs):
    params = {
        'title':None,
        'xlabel':"x",
        'ylabel':"y",
        'zlabel':"z",
        'xticks':[],
        'yticks':[],
        'zticks':[],
        'rotate_3dlabel':False,
        'labelsize':16,
        'ticksize':14,
    }
    params.update(kwargs)

    '''Format the figure axes. For 2D plots, zlabel and zticks parameters are None.'''
    axis.set_title(params['title'], weight='bold')
    axis.set_xticks(params['xticks'])
    axis.set_yticks(params['yticks'])
    axis.set_xlabel(params['xlabel'], fontsize=params['labelsize'])
    axis.set_ylabel(params['ylabel'], fontsize=params['labelsize'])

    '''if 3d projection plot'''
    try:
        axis.set_zticks(params['zticks'])
        axis.set_zlabel(params['zlabel'])
        axis.xaxis.set_rotate_label(params['rotate_3dlabel'])
        axis.yaxis.set_rotate_label(params['rotate_3dlabel'])
        axis.zaxis.set_rotate_label(params['rotate_3dlabel'])
    except AttributeError:
        pass
    return

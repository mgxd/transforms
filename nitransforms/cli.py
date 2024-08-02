from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
from textwrap import dedent

from nitransforms.base import TransformBase
from nitransforms.io.base import xfm_loader
from nitransforms.linear import load as linload
from nitransforms.nonlinear import load as nlinload
from nitransforms.resampling import apply

import pprint

def cli_apply(pargs):
    """
    Apply a transformation to an image, resampling on the reference.

    Sample usage:

        $ nt apply xform.fsl moving.nii.gz --ref reference.nii.gz --out moved.nii.gz

        $ nt apply warp.nii.gz moving.nii.gz --fmt afni --nonlinear

    """
    fmt = pargs.fmt or pargs.transform.split(".")[-1]
    if fmt in ("tfm", "mat", "h5", "x5"):
        fmt = "itk"
    elif fmt == "lta":
        fmt = "fs"

    if fmt not in ("fs", "itk", "fsl", "afni", "x5"):
        raise ValueError(
            "Cannot determine transformation format, manually set format with the `--fmt` flag"
        )

    xfm = (
        nlinload(pargs.transform, fmt=fmt)
        if pargs.nonlinear
        else linload(pargs.transform, fmt=fmt)
    )

    # ensure a reference is set
    xfm.reference = pargs.ref or pargs.moving

    moved = apply(
        xfm,
        pargs.moving,
        order=pargs.order,
        mode=pargs.mode,
        cval=pargs.cval,
        prefilter=pargs.prefilter,
    )
    # moved.to_filename(pargs.out or f"nt_{os.path.basename(pargs.moving)}")


def cli_xfm_util(pargs):
    """ """

    xfm_data = xfm_loader(pargs.transform)
    xfm_x5 = TransformBase(**xfm_data)

    if pargs.info:
        pprint.pprint(xfm_x5.x5_struct)
        print(f"Shape:\n{xfm_x5._shape}")
        print(f"Affine:\n{xfm_x5._affine}")

    if pargs.x5:
        filename = f"{os.path.basename(pargs.transform).split('.')[0]}.x5"
        xfm_x5.to_filename(filename)
        print(f"Writing out {filename}")


def cli_xfm_util(pargs):
    """
    """

    xfm_data = xfm_loader(pargs.transform)
    xfm_x5 = TransformBase(**xfm_data)

    if pargs.info:
        pprint.pprint(xfm_x5.x5_struct)
        print(f"Shape:\n{xfm_x5._shape}")
        print(f"Affine:\n{xfm_x5._affine}")

    if pargs.x5:
        filename = f"{os.path.basename(pargs.transform).split('.')[0]}.x5"
        xfm_x5.to_filename(filename)
        print(f"Writing out {filename}")
        

def get_parser():
    desc = dedent(
        """
        NiTransforms command-line utility.

        Commands:

            apply       Apply a transformation to an image
            xfm_util    Assorted transform utilities

        For command specific information, use 'nt <command> -h'.
    """
    )

    parser = ArgumentParser(
        description=desc, formatter_class=RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command")

    def _add_subparser(name, description):
        subp = subparsers.add_parser(
            name,
            description=dedent(description),
            formatter_class=RawDescriptionHelpFormatter,
        )
        return subp

    applyp = _add_subparser("apply", cli_apply.__doc__)
    applyp.set_defaults(func=cli_apply)
    applyp.add_argument("transform", help="The transform file")
    applyp.add_argument("moving", help="The image containing the data to be resampled")
    applyp.add_argument("--ref", help="The reference space to resample onto")
    applyp.add_argument(
        "--fmt",
        choices=("itk", "fsl", "afni", "fs", "x5"),
        help="Format of transformation. If no option is passed, nitransforms will "
        "estimate based on the transformation file extension.",
    )
    applyp.add_argument(
        "--out", help="The transformed image. If not set, will be set to `nt_{moving}`"
    )
    applyp.add_argument(
        "--nonlinear",
        action="store_true",
        help="Transformation is nonlinear (default: False)",
    )
    applykwargs = applyp.add_argument_group("Apply customization")
    applykwargs.add_argument(
        "--order",
        type=int,
        default=3,
        choices=range(6),
        help="The order of the spline transformation (default: 3)",
    )
    applykwargs.add_argument(
        "--mode",
        choices=("constant", "reflect", "nearest", "mirror", "wrap"),
        default="constant",
        help="Determines how the input image is extended when the resampling overflows a border "
        "(default: constant)",
    )
    applykwargs.add_argument(
        "--cval",
        type=float,
        default=0.0,
        help='Constant used when using "constant" mode (default: 0.0)',
    )
    applykwargs.add_argument(
        "--prefilter",
        action="store_false",
        help="Determines if the image's data array is prefiltered with a spline filter before "
        "interpolation (default: True)",
    )

    xfm_util = _add_subparser("xfm_util", cli_xfm_util.__doc__)
    xfm_util.set_defaults(func=cli_xfm_util)
    xfm_util.add_argument("transform", help="The transform file")
    xfm_util.add_argument(
        "--info", action="store_true", help="Get information about the transform"
    )
    xfm_util.add_argument(
        "--x5", action="store_true", help="Convert transform to .x5 file format."
    )

    return parser, subparsers


def main(pargs=None):
    parser, subparsers = get_parser()
    pargs = parser.parse_args(pargs)

    try:
        pargs.func(pargs)
    except Exception as e:
        subparser = subparsers.choices[pargs.command]
        subparser.print_help()
        raise (e)


if __name__ == "__main__":
    main()

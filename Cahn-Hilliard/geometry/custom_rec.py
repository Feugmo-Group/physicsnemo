from operator import mul
from sympy import (
    Symbol,
    Abs,
    Max,
    Min,
    sqrt,
    sin,
    cos,
    acos,
    atan2,
    pi,
    Heaviside,
    Piecewise,
)
from functools import reduce

pi = float(pi)
from sympy.vector import CoordSys3D
from .curve import SympyCurve
from .helper import _sympy_sdf_to_sdf
from .geometry import Geometry, csg_curve_naming
from .parameterization import Parameterization, Parameter, Bounds

import copy
import numpy as np
import itertools
import sympy
from typing import Callable, Union, List

from physicsnemo.sym.constants import diff_str
from .parameterization import Parameterization, Bounds
from .helper import (
    _concat_numpy_dict_list,
    _sympy_criteria_to_criteria,
    _sympy_func_to_func,
)
from .custom_geometry import c_Geometry



class custom_Rectangle(c_Geometry):
    """
    2D Rectangle

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of rectangle
    point_2 : tuple with 2 ints or floats
        upper bound point of rectangle
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        x, y = Symbol("x"), Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        line_1 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_1[1],
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_2 = SympyCurve(
            functions={
                "x": point_2[0],
                "y": l * dist_y + point_1[1],
                "normal_x": 1,
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        line_3 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_2[1],
                "normal_x": 0,
                "normal_y": 1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_4 = SympyCurve(
            functions={
                "x": point_1[0],
                "y": -l * dist_y + point_2[1],
                "normal_x": -1,
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        curves = [line_1, line_2, line_3, line_4]

        # calculate SDF
        center_x = point_1[0] + (dist_x) / 2
        center_y = point_1[1] + (dist_y) / 2
        x_diff = Abs(x - center_x) - (point_2[0] - center_x)
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(x_diff, 0) ** 2 + Max(y_diff, 0) ** 2)
        inside_distance = Min(Max(x_diff, y_diff), 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Rectangle
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )

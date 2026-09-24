from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import microcubed
from microcubed import Magnet
from microcubed.viz import plot_1d, plot_2d, plot_3d, sample_field


@pytest.fixture
def magnet():
    return Magnet([500, 300, 50], [0, 0, 0], [1.467e6, 0, 0])


def test_sample_field_preserves_xyz_shape(magnet):
    coordinates, field = sample_field(magnet, x=(-450, 450, 9), y=(-350, 350, 7), z=-150)
    assert tuple(len(values) for values in coordinates) == (9, 7, 1)
    assert field.shape == (3, 9, 7, 1)
    assert np.isfinite(field).all()


def test_plot_sections(magnet):
    figure_1d, axes_1d = plot_1d(magnet, x=(-450, 450, 9), y=0, z=-150, component="x")
    figure_2d, axes_2d = magnet.plot_2d(x=(-450, 450, 9), y=(-350, 350, 7), z=-150, component="z")
    figure_3d, axes_3d = plot_3d(
        magnet,
        x=(-450, 450, 4),
        y=(-350, 350, 3),
        z=(-300, -100, 2),
        max_points=24,
    )

    assert len(axes_1d.lines) == 1
    assert len(axes_2d.collections) == 1
    assert axes_3d.name == "3d"
    plt.close(figure_1d)
    plt.close(figure_2d)
    plt.close(figure_3d)


def test_gradient_component_and_dimension_validation(magnet):
    figure, axes = plot_2d(
        magnet,
        x=(-450, 450, 5),
        y=(-350, 350, 4),
        z=-150,
        what="dBfield",
        component=("x", "z"),
    )
    assert axes.get_title() == "dBfield dz/dx"
    plt.close(figure)

    with pytest.raises(ValueError, match="exactly 1"):
        plot_1d(magnet, x=(-1, 1, 3), y=(-1, 1, 3), z=0)


def test_top_level_visualization_wrappers_and_existing_axes(magnet):
    figure_1d, input_axis = plt.subplots()
    returned_figure, returned_axis = microcubed.plot_1d(
        magnet, x=(-2, 2, 3), y=0, z=-100, component=0, ax=input_axis, color="red"
    )
    assert returned_figure is figure_1d
    assert returned_axis is input_axis
    assert input_axis.lines[0].get_color() == "red"

    figure_2d, input_axis_2d = plt.subplots()
    returned_figure_2d, returned_axis_2d = microcubed.plot_2d(
        magnet,
        x=(-2, 2, 3),
        y=(-2, 2, 3),
        z=-100,
        component="magnitude",
        ax=input_axis_2d,
        colorbar=False,
    )
    assert returned_figure_2d is figure_2d
    assert returned_axis_2d is input_axis_2d
    assert len(figure_2d.axes) == 1

    figure_3d = plt.figure()
    input_axis_3d = figure_3d.add_subplot(projection="3d")
    returned_figure_3d, returned_axis_3d = microcubed.plot_3d(
        magnet,
        x=(-2, 2, 2),
        y=(-2, 2, 2),
        z=(-200, -100, 2),
        component="z",
        ax=input_axis_3d,
        max_points=2,
    )
    assert returned_figure_3d is figure_3d
    assert returned_axis_3d is input_axis_3d

    coordinates, values = microcubed.sample_field(magnet, x=(-2, 2, 2), y=0, z=-100, what="Hfield")
    assert tuple(len(axis) for axis in coordinates) == (2, 1, 1)
    assert values.shape == (3, 2, 1, 1)
    plt.close("all")


def test_magnet_plot_wrappers(magnet):
    figures = [
        magnet.plot_1d(x=(-2, 2, 3), y=0, z=-100),
        magnet.plot_2d(x=(-2, 2, 3), y=(-2, 2, 3), z=-100, colorbar=False),
        magnet.plot_3d(x=(-2, 2, 2), y=(-2, 2, 2), z=(-200, -100, 2)),
    ]
    assert [axes.name for _, axes in figures] == ["rectilinear", "rectilinear", "3d"]
    plt.close("all")


@pytest.mark.parametrize("component", ["bad", 3, -1])
def test_invalid_vector_component(component, magnet):
    with pytest.raises(ValueError, match="Unknown component|Component index"):
        plot_1d(magnet, x=(-2, 2, 3), y=0, z=-10, component=component)


def test_invalid_gradient_component_selection(magnet):
    with pytest.raises(ValueError, match="selected as a pair"):
        plot_2d(magnet, x=(-2, 2, 3), y=(-2, 2, 3), z=-10, what="dBfield", component="x")
    with pytest.raises(ValueError, match="Unknown component"):
        plot_2d(
            magnet,
            x=(-2, 2, 3),
            y=(-2, 2, 3),
            z=-10,
            what="dBfield",
            component=("bad", "x"),
        )


def test_invalid_coordinate_ranges(magnet):
    with pytest.raises(ValueError, match="non-empty"):
        sample_field(magnet, x=[], y=0, z=-10)
    with pytest.raises(ValueError, match="one-dimensional"):
        sample_field(magnet, x=[[1, 2]], y=0, z=-10)


def test_3d_validation(magnet):
    with pytest.raises(ValueError, match="vector field"):
        plot_3d(
            magnet,
            x=(-2, 2, 2),
            y=(-2, 2, 2),
            z=(-20, -10, 2),
            what="dBfield",
            component=("x", "z"),
        )
    with pytest.raises(ValueError, match="greater than zero"):
        plot_3d(magnet, x=(-2, 2, 2), y=(-2, 2, 2), z=(-20, -10, 2), max_points=0)

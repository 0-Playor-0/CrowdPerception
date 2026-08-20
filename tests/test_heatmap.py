"""Tests for perception/heatmap.py.

The key correctness property: total density * cell area must equal the
in-quad person count for a single frame (within float tolerance) -- i.e.
the heatmap is a genuine partition of "where the people are", not a
lossy or double-counting approximation. Achieved here with sigma=0 (no
smoothing) and exactly one update() call (window of 1 frame, so the
window-averaging divide-by-n_frames is a no-op) -- smoothing and temporal
averaging are separately-tested features, not confounders of the integral
check.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.heatmap import (
    ALARM_ONSET_PERSONS_PER_M2,
    CRUSH_BENCHMARK_PERSONS_PER_M2,
    DEFAULT_DENSITY_VMAX_PERSONS_PER_M2,
    COLOR_STOPS_EASED,
    COLOR_STOPS_HARD,
    GroundHeatmap,
    _grid_to_bgr,
    _piecewise_colormap,
)

RECT_QUAD_M = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]])


def test_density_integral_equals_in_quad_count_single_frame() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=1.0, window_seconds=1.0, sigma=0.0)
    points = np.array([[0.5, 0.5], [1.5, 0.5], [3.9, 5.9], [2.0, 3.0], [2.1, 3.1]])
    heatmap.update(t_sec=0.0, world_xy_in_quad=points)

    density = heatmap.grid("density")
    integral = float(density.sum()) * heatmap.cell_area_m2()

    assert integral == pytest.approx(len(points), abs=1e-9)


def test_count_grid_sums_to_point_count_single_frame() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=0.5, window_seconds=1.0, sigma=0.0)
    points = np.array([[1.0, 1.0], [1.1, 1.1], [3.0, 5.0]])
    heatmap.update(t_sec=0.0, world_xy_in_quad=points)

    count_grid = heatmap.grid("count")
    assert float(count_grid.sum()) == pytest.approx(len(points), abs=1e-9)


def test_rolling_window_averages_across_frames() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=1.0, window_seconds=2.0, sigma=0.0)
    # Same single point present every frame for 4 frames spanning 0..1.5s (all within the 2s window)
    for t in (0.0, 0.5, 1.0, 1.5):
        heatmap.update(t, np.array([[0.5, 0.5]]))

    count_grid = heatmap.grid("count")
    # A point present in every one of the n_frames_in_window frames should
    # average to ~1 occupant in its cell, not n_frames occupants.
    assert float(count_grid.sum()) == pytest.approx(1.0, abs=1e-9)
    assert heatmap.n_frames_in_window() == 4


def test_old_frames_drop_out_of_the_window() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=1.0, window_seconds=1.0, sigma=0.0)
    heatmap.update(t_sec=0.0, world_xy_in_quad=np.array([[0.5, 0.5]]))
    heatmap.update(t_sec=5.0, world_xy_in_quad=np.zeros((0, 2)))   # 5s later, well past the 1s window
    assert heatmap.n_frames_in_window() == 1   # only the t=5.0 (empty) frame remains


def test_points_outside_grid_bounds_are_dropped_not_erroring() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=1.0, window_seconds=1.0, sigma=0.0)
    # -100/-100 is nowhere near the quad's bounding box -- must not raise, must not be counted.
    heatmap.update(t_sec=0.0, world_xy_in_quad=np.array([[-100.0, -100.0], [1.0, 1.0]]))
    assert float(heatmap.grid("count").sum()) == pytest.approx(1.0, abs=1e-9)


def test_density_vmax_defaults_to_the_fixed_ceiling() -> None:
    # density mode's scale is a fixed real-world constant (4.0 persons/m^2),
    # NOT derived from count_vmax or cell_size_m -- changing either must not
    # move it.
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=0.5, count_vmax=5.0)
    assert heatmap.vmax("density") == pytest.approx(DEFAULT_DENSITY_VMAX_PERSONS_PER_M2)
    assert heatmap.vmax("density") == pytest.approx(4.0)


def test_density_vmax_override_is_independent_of_count_vmax() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=0.5, count_vmax=5.0, density_vmax=1.5)
    assert heatmap.vmax("density") == pytest.approx(1.5)
    assert heatmap.vmax("count") == pytest.approx(5.0)


def test_grid_shape_matches_quad_bbox_and_cell_size() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M, cell_size_m=1.0)
    assert heatmap.n_cols == 4   # 4m / 1m
    assert heatmap.n_rows == 6   # 6m / 1m


def test_invalid_mode_raises() -> None:
    heatmap = GroundHeatmap(RECT_QUAD_M)
    with pytest.raises(ValueError):
        heatmap.grid("not-a-real-mode")


# ---------------------------------------------------------------------------
# Colour ramp
# ---------------------------------------------------------------------------

def test_piecewise_colormap_endpoints() -> None:
    values = np.array([0.0, DEFAULT_DENSITY_VMAX_PERSONS_PER_M2])
    for stops in (COLOR_STOPS_HARD, COLOR_STOPS_EASED):
        colors = _piecewise_colormap(values, stops)
        np.testing.assert_array_equal(colors[0], [200, 255, 255])   # 0.0 -> pale yellow
        np.testing.assert_array_equal(colors[1], [0, 0, 255])       # ceiling -> full red


def test_piecewise_colormap_clips_out_of_range_values() -> None:
    colors = _piecewise_colormap(np.array([-5.0, 999.0]), COLOR_STOPS_HARD)
    np.testing.assert_array_equal(colors[0], [200, 255, 255])   # clipped to the lowest stop
    np.testing.assert_array_equal(colors[1], [0, 0, 255])       # clipped to the highest stop


@pytest.mark.parametrize("stops", [COLOR_STOPS_HARD, COLOR_STOPS_EASED])
def test_alarm_onset_reads_as_clearly_red_in_both_variants(stops) -> None:
    """Required behaviour from the task: AT the alarm onset (2.5 persons/m^2)
    the colour must already be clearly red (low green, saturated red) in
    BOTH the hard and eased variants -- they only differ in how the ramp
    APPROACHES that point, not in what colour it lands on there."""
    color = _piecewise_colormap(np.array([ALARM_ONSET_PERSONS_PER_M2]), stops)[0]
    blue, green, red = int(color[0]), int(color[1]), int(color[2])
    assert red == 255
    assert green < 60, f"green={green} is too high to read as 'clearly red' at the alarm onset"
    assert blue == 0


def test_hard_variant_transitions_more_steeply_than_eased_near_the_alarm_onset() -> None:
    """The whole point of the two variants: a cell's colour, sampled just
    below vs. just above the alarm onset, should jump much more for the
    hard ramp than for the eased one -- that bigger per-unit-density colour
    change IS what "strobes more" and "sharper edge" mean here."""
    just_below = ALARM_ONSET_PERSONS_PER_M2 - 0.1
    just_above = ALARM_ONSET_PERSONS_PER_M2 + 0.1

    hard_below, hard_above = _piecewise_colormap(np.array([just_below, just_above]), COLOR_STOPS_HARD)
    eased_below, eased_above = _piecewise_colormap(np.array([just_below, just_above]), COLOR_STOPS_EASED)

    hard_jump = float(np.linalg.norm(hard_above.astype(float) - hard_below.astype(float)))
    eased_jump = float(np.linalg.norm(eased_above.astype(float) - eased_below.astype(float)))
    assert hard_jump > eased_jump, (
        f"expected a bigger colour jump across the same +/-0.1 persons/m^2 window straddling "
        f"the alarm onset for the hard variant (got hard={hard_jump:.1f}, eased={eased_jump:.1f})"
    )


def test_crush_benchmark_is_past_the_alarm_onset() -> None:
    # Sanity-check the two named constants are in the order the docs/legend assume.
    assert 0.0 < ALARM_ONSET_PERSONS_PER_M2 < CRUSH_BENCHMARK_PERSONS_PER_M2 < DEFAULT_DENSITY_VMAX_PERSONS_PER_M2


def test_grid_to_bgr_rescales_non_density_vmax_onto_the_same_ramp() -> None:
    # A grid whose OWN vmax is 1.0 (e.g. the pixel-space overlay) should map
    # its own maximum value onto the same "full red" colour the density
    # panel's 4.0 maps to -- same ramp shape, different native scale.
    grid = np.array([[1.0]])
    color_at_own_vmax = _grid_to_bgr(grid, vmax=1.0, color_variant="hard")[0, 0]
    np.testing.assert_array_equal(color_at_own_vmax, [0, 0, 255])

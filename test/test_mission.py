"""
Tests for the pure module-level mission helpers in web_control_node:
``map_pixel_to_world``, ``deposit_serialize`` and ``deposit_parse``.

Import strategy: the helpers have no ROS or aiohttp dependency, so the module
imports without a running ROS context (rclpy.init() is never called at import
time).  The ROS / aiohttp stubbing for bare environments and the sys.path
setup live in the shared ``conftest.py`` (imported by pytest before this
module).
"""

import importlib

import pytest


# ---------------------------------------------------------------------------
# Import the module under test (stubs installed by conftest.py)
# ---------------------------------------------------------------------------

try:
    _mod = importlib.import_module('jetank_web_control.web_control_node')
    map_pixel_to_world = _mod.map_pixel_to_world
    deposit_serialize = _mod.deposit_serialize
    deposit_parse = _mod.deposit_parse
    is_terminal_mission_status = _mod.is_terminal_mission_status
except Exception as exc:  # noqa: BLE001
    pytest.skip(f'Could not import web_control_node: {exc}',
                allow_module_level=True)


# ---------------------------------------------------------------------------
# map_pixel_to_world
# ---------------------------------------------------------------------------

# A simple known map: 0.05 m/px, 10x8 cells, origin at (-1.0, -2.0).
RES = 0.05
W = 10
H = 8
OX = -1.0
OY = -2.0


def _w(ix, iy):
    return map_pixel_to_world(ix, iy, RES, W, H, OX, OY)


class TestMapPixelToWorld:
    def test_top_left_pixel(self):
        # PNG (0,0) is the top-left == the *top* occupancy row (H-1).
        wx, wy = _w(0, 0)
        assert wx == pytest.approx(OX + 0.5 * RES)            # -0.975
        assert wy == pytest.approx(OY + (H - 1 + 0.5) * RES)  # -1.625

    def test_bottom_left_pixel(self):
        # PNG bottom-left maps to occupancy grid row 0 (the origin row).
        wx, wy = _w(0, H - 1)
        assert wx == pytest.approx(OX + 0.5 * RES)            # -0.975
        assert wy == pytest.approx(OY + 0.5 * RES)            # -1.975

    def test_top_right_pixel(self):
        wx, wy = _w(W - 1, 0)
        assert wx == pytest.approx(OX + (W - 1 + 0.5) * RES)  # -0.525
        assert wy == pytest.approx(OY + (H - 1 + 0.5) * RES)  # -1.625

    def test_bottom_right_pixel(self):
        wx, wy = _w(W - 1, H - 1)
        assert wx == pytest.approx(OX + (W - 1 + 0.5) * RES)  # -0.525
        assert wy == pytest.approx(OY + 0.5 * RES)            # -1.975

    def test_interior_pixel(self):
        # pixel (4, 2): grid_row = 7 - 2 = 5
        wx, wy = _w(4, 2)
        assert wx == pytest.approx(OX + (4 + 0.5) * RES)  # -0.775
        assert wy == pytest.approx(OY + (5 + 0.5) * RES)  # -1.725

    def test_vertical_flip_is_applied(self):
        # Two pixels mirrored across the vertical centre differ only in wy,
        # and their wy values straddle the map centre symmetrically.
        _, wy_top = _w(3, 1)
        _, wy_bot = _w(3, H - 2)
        centre = OY + (H / 2.0) * RES
        assert wy_top + wy_bot == pytest.approx(2 * centre)

    def test_clamp_negative(self):
        # Negative pixels clamp to the (0,0) edge cell.
        assert _w(-5, -5) == _w(0, 0)

    def test_clamp_overflow(self):
        # Pixels past the far edge clamp to (W-1, H-1).
        assert _w(W + 50, H + 50) == _w(W - 1, H - 1)

    def test_float_pixels_coerced_to_int(self):
        # Floats are truncated toward zero by int(), matching navigate_to_pixel.
        assert _w(4.9, 2.9) == _w(4, 2)

    def test_returns_plain_floats(self):
        wx, wy = _w(1, 1)
        assert isinstance(wx, float) and isinstance(wy, float)

    def test_zero_origin(self):
        wx, wy = map_pixel_to_world(0, H - 1, RES, W, H, 0.0, 0.0)
        assert wx == pytest.approx(0.5 * RES)
        assert wy == pytest.approx(0.5 * RES)


# ---------------------------------------------------------------------------
# deposit_serialize / deposit_parse
# ---------------------------------------------------------------------------

class TestDepositSerialize:
    def test_roundtrip(self):
        text = deposit_serialize(1.25, -3.5)
        assert deposit_parse(text) == (1.25, -3.5)

    def test_serialize_coerces_to_float(self):
        # ints in -> floats stored -> floats parsed
        assert deposit_parse(deposit_serialize(2, 3)) == (2.0, 3.0)

    def test_serialize_is_valid_json(self):
        import json
        d = json.loads(deposit_serialize(0.1, 0.2))
        assert d == {'x': 0.1, 'y': 0.2}


class TestDepositParse:
    def test_valid(self):
        assert deposit_parse('{"x": 4.0, "y": 5.0}') == (4.0, 5.0)

    def test_int_values_accepted(self):
        assert deposit_parse('{"x": 4, "y": 5}') == (4.0, 5.0)

    def test_empty_string_returns_none(self):
        assert deposit_parse('') is None

    def test_whitespace_returns_none(self):
        assert deposit_parse('   \n ') is None

    def test_malformed_json_returns_none(self):
        assert deposit_parse('{not json') is None

    def test_non_object_returns_none(self):
        assert deposit_parse('[1, 2]') is None
        assert deposit_parse('42') is None

    def test_missing_key_returns_none(self):
        assert deposit_parse('{"x": 1.0}') is None

    def test_non_numeric_value_returns_none(self):
        assert deposit_parse('{"x": "a", "y": 2.0}') is None

    def test_none_input_returns_none(self):
        assert deposit_parse(None) is None


# ---------------------------------------------------------------------------
# is_terminal_mission_status
# ---------------------------------------------------------------------------

class TestIsTerminalMissionStatus:
    def test_done_is_terminal(self):
        assert is_terminal_mission_status('DONE') is True

    def test_failed_is_terminal(self):
        assert is_terminal_mission_status('FAILED') is True

    def test_cancelled_is_terminal(self):
        assert is_terminal_mission_status('CANCELLED') is True

    def test_idle_is_terminal(self):
        assert is_terminal_mission_status('IDLE') is True

    def test_in_progress_states_not_terminal(self):
        for s in ('NAVIGATE_TO_SITE', 'SEARCH', 'PICK',
                  'NAVIGATE_TO_DEPOSIT', 'DEPOSIT'):
            assert is_terminal_mission_status(s) is False

    def test_case_insensitive(self):
        assert is_terminal_mission_status('done') is True
        assert is_terminal_mission_status('search') is False

    def test_trailing_text_still_terminal(self):
        # The coordinator/UI may append a reason after the state token.
        assert is_terminal_mission_status('FAILED — no sock found') is True
        assert is_terminal_mission_status('DONE ✓') is True

    def test_empty_and_none_are_terminal(self):
        assert is_terminal_mission_status('') is True
        assert is_terminal_mission_status('   ') is True
        assert is_terminal_mission_status(None) is True

    def test_non_string_is_terminal(self):
        assert is_terminal_mission_status(42) is True
        assert is_terminal_mission_status(['SEARCH']) is True

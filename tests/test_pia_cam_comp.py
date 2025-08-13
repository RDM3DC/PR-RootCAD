from adaptivecad.cam.pia_compensation import PiAParams, compensate_gcode_lines

def test_compensates_r_mode_arc():
    params = PiAParams(beta=0.2, s0=1.0, clamp=0.3)
    lines = ["G2 X10.0 Y5.0 R2.000 F300\n"]
    out = compensate_gcode_lines(lines, params, min_arc=0.1, max_arc=100.0)
    assert "R2.000000" not in out[0]  # radius should change
    assert out[0].startswith("G2")

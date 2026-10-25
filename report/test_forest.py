import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(__file__))
import forest as F


def fixture(tmp_path):
    vs = {"prompt:capability": {"difference": -1, "interval95": [-3.0, 0.0]},
          "calibration (nested)": {"difference": 1, "interval95": [-2.0, 4.0]},
          "other": {"difference": 9, "interval95": [0.0, 9.0]}}
    p = tmp_path / "iv.json"
    p.write_text(json.dumps({"vs_majority": vs}))
    return p


def test_rows_keep_table_labels_and_drop_unknown_keys(tmp_path):
    assert F.rows(fixture(tmp_path)) == [("Prompt-only: capability", -1, -3.0, 0.0),
                                         ("Calibration (nested)", 1, -2.0, 4.0)]


def test_plot_writes_a_wide_short_png(tmp_path):
    out = tmp_path / "f.png"
    F.plot(F.rows(fixture(tmp_path)), out)
    head = out.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", head[16:24])
    assert w == 1200 and 380 <= h <= 420

import sys
from pathlib import Path
from unittest.mock import patch

import ase.io
import numpy as np
import pytest

from twistHS.lib.coincidence import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_MAX_ATOMS,
    DEFAULT_MOIRE_ORDER,
    DEFAULT_MOIRE_TOL,
    DEFAULT_TOLERANCE,
)
from twistHS.lib.workflow import export_structure
from twistHS.twist_cli import main
from twistHS.twist_gui import preview_supercell, scan_bilayer

DATA = Path(__file__).parent
GRAPHENE = str(DATA / "graphene.xsf")
MOS2 = str(DATA / "MoS2.xsf")


def gui_scan(bottom, top, start, stop, step, *, method="search", tolerance=DEFAULT_TOLERANCE,
             angle_tol=DEFAULT_ANGLE_TOL, moire_tol=DEFAULT_MOIRE_TOL, moire_order=DEFAULT_MOIRE_ORDER,
             max_atoms=DEFAULT_MAX_ATOMS, max_order=None):
    return scan_bilayer(bottom, top, start, stop, step, 30, 7, tolerance, method,
                        angle_tol, moire_tol, moire_order, max_atoms, "numpy", max_order)


@pytest.mark.parametrize("method", ["search", "first-order"])
def test_interfaces_return_identical_science(monkeypatch, capsys, method):
    entries, rows = gui_scan(GRAPHENE, GRAPHENE, 21.84, 21.8402, 0.0001, method=method,
                             tolerance=0.5, moire_tol=0.3, max_order=2)
    assert [entry.requested_angle for entry in entries] == [21.84, 21.8401]
    monkeypatch.setattr(sys, "argv", [
        "twist_cli", "-b", GRAPHENE, "-t", GRAPHENE, "-al", "21.84", "21.8402", "0.0001",
        "--method", method, "-tol", "0.5", "--moire-tol", "0.3", "-mo", "2",
    ])
    with (patch("twistHS.twist_cli.StructureVisualizer") as visualizer,
          patch("matplotlib.pyplot.show")):
        main()
    cli_entries = visualizer.call_args.args[0]
    for gui, cli in zip(entries, cli_entries, strict=True):
        assert (gui.requested_angle, gui.effective_angle, gui.strain) == (
            cli.requested_angle, cli.effective_angle, cli.strain)
        np.testing.assert_array_equal(gui.atoms.numbers, cli.atoms.numbers)
        np.testing.assert_array_equal(gui.atoms.cell, cli.atoms.cell)
        np.testing.assert_array_equal(gui.atoms.positions, cli.atoms.positions)
    assert capsys.readouterr().out.splitlines()[2:] == ["\t".join(row) for row in rows]


def test_no_matching_supercell_still_returns_moire_components():
    entries, rows = gui_scan(GRAPHENE, MOS2, 0, 1, 1, max_atoms=1000, moire_order=4, max_order=2)
    assert not entries
    assert rows
    assert {row[0] for row in rows} == {"0.0"}


def test_preview_and_export_use_stored_result(monkeypatch, tmp_path):
    entries, _ = gui_scan(GRAPHENE, GRAPHENE, 21.84, 21.8402, 0.0001, moire_tol=0.3)
    assert entries[0].effective_angle == pytest.approx(entries[1].effective_angle)
    selected = entries[1]
    monkeypatch.setattr("twistHS.twist_gui.run_scan", lambda *args, **kwargs: pytest.fail("Unexpected rescan"))
    figure, _, _ = preview_supercell(1, entries)
    assert selected.label in figure.axes[0].get_title()

    default = export_structure(selected, directory=tmp_path)
    assert default.name == "structure_21.84.xsf"
    custom = export_structure(selected, "bilayer.traj", directory=tmp_path)
    explicit = export_structure(selected, "explicit.data", "extxyz", directory=tmp_path)
    for path, format in ((default, None), (custom, None), (explicit, "extxyz")):
        saved = ase.io.read(path, format=format)
        np.testing.assert_allclose(saved.cell, selected.atoms.cell, rtol=0, atol=1e-7)
        np.testing.assert_allclose(saved.positions, selected.atoms.positions, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="Specify a filename"):
        export_structure(selected, outformat="xyz", directory=tmp_path)

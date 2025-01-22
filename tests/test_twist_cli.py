import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ase.io
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
import pytest

from twistHS.lib.workflow import NO_SUPERCELLS, Supercell, angle_range
from twistHS.twist_cli import StructureVisualizer, main

DATA = Path(__file__).parent
GRAPHENE = str(DATA / "graphene.xsf")
MOS2 = str(DATA / "MoS2.xsf")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("start, stop, step, expected", [
    ("21.84", "21.8402", "0.0001", [21.84, 21.8401]),
    (0.3, 0, -0.1, [0.3, 0.2, 0.1]),
    (0, 0.25, 0.1, [0, 0.1, 0.2]),
])
def test_half_open_angle_range(start, stop, step, expected):
    assert angle_range(start, stop, step) == expected


@pytest.mark.parametrize("interval", [(0, 1, 0), (0, 1, -1), (1, 1, 1), (0, 1, "bad")])
def test_invalid_angle_range(interval):
    with pytest.raises(ValueError, match="interval"):
        angle_range(*interval)


@pytest.mark.parametrize("options, message", [
    ([], "one of the arguments"),
    (["-a", "1", "-al", "0", "2", "1"], "not allowed with argument"),
    (["-al", "0", "1"], "expected 3 arguments"),
    (["-a", "1", "-o", "xyz"], "--outformat requires --write"),
])
def test_invalid_cli_arguments(monkeypatch, capsys, options, message):
    monkeypatch.setattr(sys, "argv", ["twist_cli", "-b", GRAPHENE, "-t", GRAPHENE, *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def run_cli(monkeypatch, *options, upper=GRAPHENE):
    monkeypatch.setattr(sys, "argv", ["twist_cli", "-b", GRAPHENE, "-t", upper, *options])
    with patch("twistHS.twist_cli.StructureVisualizer", wraps=StructureVisualizer) as visualizer:
        with patch("matplotlib.pyplot.show"):
            main()
    return visualizer.call_args.args[0]


def test_scan_filters_angles_and_keeps_result_metrics_aligned(monkeypatch):
    entries = run_cli(monkeypatch, "-a", "0.05", "21.84", "5", "-tol", "0.5",
                      "--moire-tol", "0.3", "--max-atoms", "28")
    assert [entry.requested_angle for entry in entries] == [21.84]
    assert len(entries[0].atoms) == 28
    assert entries[0].strain * 100 <= 0.5
    np.testing.assert_allclose(entries[0].effective_angle, 21.7867893, rtol=0, atol=1e-7)


def test_component_table_and_requested_angle_order(monkeypatch, capsys):
    entries = run_cli(monkeypatch, "-a", "38.2132", "21.7868", "-mo", "2")
    assert [entry.requested_angle for entry in entries] == [38.2132, 21.7868]
    assert [len(entry.atoms) for entry in entries] == [28, 28]
    table = capsys.readouterr().out
    assert "Moiré wave vector (Å⁻¹)" in table
    assert "Multiplicity" in table
    assert table.index("38.2132\t") < table.index("21.7868\t")


def test_high_index_search_constraint_is_independent_of_analysis_cutoff(monkeypatch, capsys):
    options = ("-a", "0", "--max-atoms", "1000", "--max-order", "1")
    entries = run_cli(monkeypatch, *options, upper=MOS2)
    assert [len(entry.atoms) for entry in entries] == [755]
    old_analysis = capsys.readouterr().out
    with pytest.raises(SystemExit, match=NO_SUPERCELLS):
        run_cli(monkeypatch, *options, "--moire-order", "4", upper=MOS2)
    assert capsys.readouterr().out == old_analysis


def test_picking_overlapping_points_exports_the_selected_supercell(monkeypatch, tmp_path):
    first = ase.io.read(GRAPHENE)
    second = first.copy()
    second.positions[0, 2] += 0.5
    for atoms in (first, second):
        atoms.info["theta_eff"] = 21.7868
    entries = [Supercell(21.84, first, 0.002), Supercell(21.8401, second, 0.002)]
    monkeypatch.chdir(tmp_path)
    handler = StructureVisualizer(entries, None, None)
    np.testing.assert_allclose(handler.scatter.get_offsets(), [[21.7868, 2], [21.7868, 2]])
    selectors = []

    def capture_choices(*args):
        choices = RadioButtons(*args)
        selectors.append(choices)
        return choices

    with (patch("twistHS.twist_cli.Button") as button,
          patch("twistHS.twist_cli.RadioButtons", side_effect=capture_choices),
          patch("matplotlib.figure.Figure.show")):
        handler.onpick(SimpleNamespace(artist=object(), ind=[0]))
        button.assert_not_called()
        handler.onpick(SimpleNamespace(artist=handler.scatter, ind=[0, 1]))
    selectors[0].set_active(1)
    button.return_value.on_clicked.call_args.args[0](None)
    saved = ase.io.read(tmp_path / "structure_21.84.xsf")
    np.testing.assert_allclose(saved.positions, second.positions)

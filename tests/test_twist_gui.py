from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import matplotlib.pyplot as plt

from twistHS.lib.algorithm import HSGenerator
from twistHS.twist_gui import StructureVisualizer, main_interface


def test_filtered_angles_stay_aligned(tmp_path):
    root = Path(__file__).parent
    bottom = ase.io.read(root / 'graphene.xsf')
    top = bottom.copy()
    handler = StructureVisualizer(bottom, top, 30, 7)
    figure, angles, results = handler.generate_initial_plot(21.84, 21.8402, 0.0001, 0.5, moire_tol=0.3)
    np.testing.assert_array_equal(angles, [21.84, 21.8401])
    np.testing.assert_allclose(figure.axes[0].collections[0].get_offsets()[:, 0],
                               [21.7867893, 21.7867893], rtol=0, atol=1e-7)
    generator = HSGenerator(bottom, top, moire_tol=0.3)
    assert len(angles) == len(results)
    for angle, (result, strain) in zip(angles, results):
        expected, expected_strain, theta_eff = generator.gen_supercell(angle, 30, 7)
        assert len(result) == len(expected)
        assert strain == expected_strain
        assert result.info["theta_eff"] == theta_eff
    detailed = handler.generate_detailed_structure(angles[0], angles, results)
    assert isinstance(detailed, plt.Figure)
    destination = tmp_path / 'selected.xsf'
    handler.save_structure(21.8401, angles, results, str(destination))
    saved = ase.io.read(destination)
    assert len(saved) == len(results[1][0])
    np.testing.assert_allclose(saved.cell, results[1][0].cell, rtol=0, atol=1e-12)
    plt.close(figure)
    plt.close(detailed)


def test_no_results():
    root = Path(__file__).parent
    bottom = ase.io.read(root / 'graphene.xsf')
    top = ase.io.read(root / 'MoS2.xsf')
    handler = StructureVisualizer(bottom, top, 30, 7)
    assert handler.generate_initial_plot(5, 6, 1, 0) == (None, None, None)


def test_gui_and_python_api_apply_moire_order():
    root = Path(__file__).parent
    bottom, top = [SimpleNamespace(name=str(root / name)) for name in ('graphene.xsf', 'MoS2.xsf')]
    generator = HSGenerator(ase.io.read(bottom.name), ase.io.read(top.name), moire_order=4)
    protected, _, theta_eff = generator.gen_supercell(0, 30, 7)
    assert len(protected) == 1027

    _, figure, _, _ = main_interface(bottom, top, 0, 1, 1, 30, 7, 1, None, False, None, moire_order=4)
    np.testing.assert_allclose(figure.axes[0].collections[0].get_offsets(), [[theta_eff, len(protected)]])
    plt.close(figure)

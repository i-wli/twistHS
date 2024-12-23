from pathlib import Path

import ase.io
import numpy as np
import matplotlib.pyplot as plt

from twistHS.lib.algorithm import HSGenerator
from twistHS.twist_gui import StructureVisualizer


def test_filtered_angles_stay_aligned(tmp_path):
    root = Path(__file__).parent
    bottom = ase.io.read(root / 'graphene.xsf')
    top = ase.io.read(root / 'MoS2.xsf')
    handler = StructureVisualizer(bottom, top, 30, 7)
    figure, angles, results = handler.generate_initial_plot(0, 10, 1, 0.5)
    np.testing.assert_array_equal(angles, np.arange(2, 9))
    generator = HSGenerator(bottom, top)
    assert len(angles) == len(results)
    for angle, (result, strain) in zip(angles, results):
        expected, expected_strain = generator.gen_supercell(angle, 30, 7)
        assert len(result) == len(expected)
        assert strain == expected_strain
    detailed = handler.generate_detailed_structure(angles[0], angles, results)
    assert isinstance(detailed, plt.Figure)
    destination = tmp_path / 'selected.xsf'
    handler.save_structure(angles[0], angles, results, str(destination))
    assert len(ase.io.read(destination)) == len(results[0][0])
    plt.close(figure)
    plt.close(detailed)


def test_no_results():
    root = Path(__file__).parent
    bottom = ase.io.read(root / 'graphene.xsf')
    top = ase.io.read(root / 'MoS2.xsf')
    handler = StructureVisualizer(bottom, top, 30, 7)
    assert handler.generate_initial_plot(0, 10, 1, 0) == (None, None, None)

import os
import sys
from pathlib import Path

import pytest
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from unittest.mock import Mock, patch
from ase import Atoms
from twistHS.twist_cli import StructureVisualizer

@pytest.fixture
def mock_args():
    """Create mock arguments"""
    args = Mock()
    args.write = None
    args.outformat = None
    args.z = 30.0
    args.d = 7.0
    return args

@pytest.fixture
def test_structures():
    """Create test structures"""
    a = 2.46 # graphene
    bottom = Atoms('C2',
                  positions=[[0, 0, 0],
                           [a/2, a*np.sqrt(3)/2, 0]],
                  cell=[[a, 0, 0],
                        [-a/2, a*np.sqrt(3)/2, 0],
                        [0, 0, 15]])
    
    a = 2.504 # hBN
    top = Atoms(['B', 'N'],
                positions=[[0, 0, 0],
                         [a/2, a*np.sqrt(3)/2, 0]],
                cell=[[a, 0, 0],
                      [-a/2, a*np.sqrt(3)/2, 0],
                      [0, 0, 15]])
    return bottom, top

@pytest.fixture
def handler(test_structures, mock_args):
    bottom, top = test_structures
    angles = [0.0, 1.0, 2.0]
    Delta = 0.01
    results = [(bottom.copy(), Delta) for _ in range(3)]
    for angle, (structure, _) in zip(angles, results):
        structure.info["theta_eff"] = angle
    return StructureVisualizer(angles, bottom, top, mock_args, results)

def test_initialization(handler):
    """Test StructureVisualizer initialization"""
    assert handler.current_structure is None
    assert handler.ang == 0
    assert handler.num == 0
    assert handler.Delta == 0
    assert len(handler.angles) == 3
    assert len(handler.results) == 3

def test_get_filename_default(handler):
    """Test default filename generation"""
    handler.ang = 1.5
    handler.num = 100
    filename = handler.get_filename()
    assert filename == "Super_1.50_100.xsf"

def test_get_filename_custom(handler, mock_args):
    """Test custom filename"""
    mock_args.write = "custom.xsf"
    filename = handler.get_filename()
    assert filename == "custom.xsf"

@patch('matplotlib.pyplot.show')
def test_onpick_event_pathCol(mock_show, handler):
    """Test onpick event handling with PathCollection"""
    from matplotlib.collections import PathCollection

    event = Mock()
    event.artist = PathCollection([])
    event.ind = [0]

    handler.onpick(event)
    assert handler.ang == handler.angles[0]
    assert handler.Delta == handler.results[0][1]

def test_onpick_event_wrong_artist(handler):
    """Test onpick event handling with wrong artist type"""
    event = Mock()
    event.artist = Mock()
    event.ind = [0]
    
    initial_ang = handler.ang
    initial_delta = handler.Delta
    
    handler.onpick(event)
    assert handler.ang == initial_ang
    assert handler.Delta == initial_delta

def test_save_structure(handler):
    """Test structure saving"""
    with tempfile.TemporaryDirectory() as tmpdir:
        handler.args.write = os.path.join(tmpdir, "test.xsf")
        handler.current_result = handler.results[0][0]
        
        event = Mock()
        handler.save_structure(event)
        
        assert os.path.exists(handler.args.write)


def test_main_filters_angles_and_results(monkeypatch):
    from twistHS.twist_cli import main
    root = Path(__file__).parent
    monkeypatch.setattr(sys, 'argv', ['twist_cli', '-b', str(root / 'graphene.xsf'),
                        '-t', str(root / 'graphene.xsf'), '-a', '0.05', '21.84', '5', '-tol', '0.5',
                        '--angle-tol', '0.1', '--moire-tol', '0.3', '--max-atoms', '28'])
    with (patch('twistHS.twist_cli.StructureVisualizer') as visualizer,
          patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.scatter', wraps=plt.scatter) as scatter):
        main()
    angles, _, _, _, results = visualizer.call_args.args
    np.testing.assert_array_equal(angles, [21.84])
    assert len(results) == 1
    assert len(results[0][0]) == 28
    assert results[0][1] * 100 <= 0.5
    np.testing.assert_allclose(scatter.call_args.args[0], [21.7867893], rtol=0, atol=1e-7)


def test_main_applies_moire_tolerance(monkeypatch):
    from twistHS.twist_cli import main
    graphene = str(Path(__file__).parent / 'graphene.xsf')
    monkeypatch.setattr(sys, 'argv', ['twist_cli', '-b', graphene, '-t', graphene,
                        '-a', '21.84', '--moire-tol', '0.2', '--max-atoms', '28'])
    with pytest.raises(SystemExit, match='No structure within the configured limits'):
        main()


def test_main_applies_moire_order_independently_of_analysis_cutoff(monkeypatch):
    from twistHS.twist_cli import main
    root = Path(__file__).parent
    monkeypatch.setattr(sys, 'argv', ['twist_cli', '-b', str(root / 'graphene.xsf'),
                        '-t', str(root / 'MoS2.xsf'), '-a', '0', '--max-atoms', '1000', '--max-order', '1'])
    with patch('twistHS.twist_cli.StructureVisualizer') as visualizer, patch('matplotlib.pyplot.show'):
        main()
    _, _, _, _, results = visualizer.call_args.args
    assert [len(result) for result, _ in results] == [755]

    monkeypatch.setattr(sys, 'argv', sys.argv + ['--moire-order', '4.0'])
    with pytest.raises(SystemExit, match='No structure within the configured limits'):
        main()


def test_main_no_structure_within_tolerance(monkeypatch):
    from twistHS.twist_cli import main
    root = Path(__file__).parent
    monkeypatch.setattr(sys, 'argv', ['twist_cli', '-b', str(root / 'graphene.xsf'),
                        '-t', str(root / 'MoS2.xsf'), '-a', '0', '-tol', '0', '--max-atoms', '5'])
    with pytest.raises(SystemExit, match='No structure within the configured limits'):
        main()


def test_main_requires_angles(monkeypatch, capsys):
    from twistHS.twist_cli import main

    monkeypatch.setattr(sys, 'argv', ['twist_cli', '-b', 'bottom.xsf', '-t', 'top.xsf'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert 'specify at least one of -a/--angle or -al/--alist' in capsys.readouterr().err


def test_main_component_table_and_commensurate_cells(monkeypatch, capsys):
    from twistHS.twist_cli import main
    root = Path(__file__).parent
    monkeypatch.setattr(sys, 'argv', ['twist_cli', '-b', str(root / 'graphene.xsf'),
                        '-t', str(root / 'graphene.xsf'), '-a', '21.7868', '38.2132', '-mo', '2'])
    with patch('twistHS.twist_cli.StructureVisualizer') as visualizer, patch('matplotlib.pyplot.show'):
        main()
    angles, _, _, _, results = visualizer.call_args.args
    np.testing.assert_array_equal(angles, [21.7868, 38.2132])
    assert [len(result) for result, _ in results] == [28, 28]
    table = capsys.readouterr().out
    assert table.count('Moiré components at') == 2
    assert 'multiplicity' in table
    assert 'q  period  angle (deg)' in table


def test_save_respects_explicit_format(handler, tmp_path):
    destination = tmp_path / 'structure.data'
    handler.args.write = str(destination)
    handler.args.outformat = 'xyz'
    handler.current_result = handler.results[0][0]
    handler.save_structure(None)
    assert destination.read_text().splitlines()[0] == '2'

if __name__ == '__main__':
    pytest.main(['-v'])

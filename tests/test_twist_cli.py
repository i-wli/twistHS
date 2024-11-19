import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, patch
from ase import Atoms
from twist_cli import StructureHandler

@pytest.fixture
def mock_args():
    """Create mock arguments"""
    args = Mock()
    args.write = None
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
    return StructureHandler(angles, bottom, top, mock_args, results)

def test_initialization(handler):
    """Test StructureHandler initialization"""
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

if __name__ == '__main__':
    pytest.main(['-v'])
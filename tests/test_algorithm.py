import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from twistHS.lib.algorithm import HSGenerator, check_vectors

@pytest.fixture
def test_structures():
    """Create simple test structures for bottom and top layers"""
    a = 2.46  # graphene lattice constant
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

def test_gen_supercell_basic(test_structures):
    """Test basic functionality with valid inputs"""
    bottom, top = test_structures
    theta = 5.0  # degrees
    super_z = 15.0  # Å
    delta_z = 3.4  # Å
    
    twistSupercell = HSGenerator(bottom, top)
    result, delta = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    assert isinstance(result, Atoms)
    assert isinstance(delta, float)
    assert delta >= 0
    assert len(result) > len(bottom) + len(top)
    assert result.cell[2][2] == pytest.approx(super_z, rel=1e-5)

def test_gen_supercell_layer_separation(test_structures):
    """Test if layers are properly separated by delta_z"""
    bottom, top = test_structures
    theta = 0.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top)
    result, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    # Get z-coordinates
    z_coords = result.positions[:, 2]
    unique_z = np.unique(z_coords)
    
    assert len(unique_z) == 2  # Should have exactly 2 distinct z-levels
    assert np.abs(unique_z[1] - unique_z[0]) == pytest.approx(delta_z, rel=1e-5)

def test_gen_supercell_rotation(test_structures):
    """Test if supercell is properly generated"""
    bottom, top = test_structures
    theta = 3.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top)
    result, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    # Number of atoms should increase - superlattice is generated
    assert len(result) > len(bottom) + len(top)

def test_gen_supercell_different_angles():
    """Test error handling when layers have different angles"""
    # Create bottom layer with 60° angle
    bottom = Atoms('C2',
                  positions=[[0, 0, 0], [1, 0, 0]],
                  cell=[[2, 0, 0],
                        [1, np.sqrt(3), 0],
                        [0, 0, 15]])
    
    # Create top layer with different angle
    top = Atoms('C2',
                positions=[[0, 0, 0], [1, 0, 0]],
                cell=[[2, 0, 0],
                      [1, 2, 0],
                      [0, 0, 15]])
    
    with pytest.raises(ValueError, match="The angles of two layers are different"):
        twistSupercell = HSGenerator(bottom, top)
        twistSupercell.gen_supercell(5.0, 15.0, 3.4)

def test_gen_supercell_zero_angle(test_structures):
    """Test behavior with zero rotation angle"""
    bottom, top = test_structures
    theta = 0.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top)
    result, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)

    assert result.cell.cellpar()[3] == pytest.approx(bottom.cell.cellpar()[3], rel=1e-5)

if __name__ == "__main__":
    pytest.main(["-v", "test_algorithm.py"])
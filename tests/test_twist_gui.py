import numpy as np
import matplotlib.pyplot as plt
import ase.io
from twist_gui import StructureHandler

def test_generate_initial_plot():
    # Test case 1: No results within tolerance
    handler = StructureHandler(bottom, top, z, d)
    fig, angles, results = handler.generate_initial_plot(0, 10, 1, 0.01)
    assert fig is None
    assert angles is None
    assert results is None

    # Test case 2: Results within tolerance
    handler = StructureHandler(bottom, top, z, d)
    fig, angles, results = handler.generate_initial_plot(0, 10, 1, 10)
    assert fig is not None
    assert angles is not None
    assert results is not None

def test_generate_detailed_structure():
    # Test case 1: Angle within generated range
    handler = StructureHandler(bottom, top, z, d, tolerance)
    fig = handler.generate_detailed_structure(5, angles, results)
    assert fig is not None

    # Test case 2: Angle not within generated range
    handler = StructureHandler(bottom, top, z, d, tolerance)
    fig = handler.generate_detailed_structure(15, angles, results)
    assert fig == "Angle not in the generated range. Please enter a valid angle."

def test_save_structure():
    # Test case 1: Angle within generated range
    handler = StructureHandler(bottom, top, z, d, tolerance)
    filename = handler.save_structure(5, angles, results, "test.xsf")
    assert filename == "test.xsf"

    # Test case 2: Angle not within generated range
    handler = StructureHandler(bottom, top, z, d, tolerance)
    filename = handler.save_structure(15, angles, results, "test.xsf")
    assert filename is None

# Run the tests
test_generate_initial_plot()
test_generate_detailed_structure()
test_save_structure()
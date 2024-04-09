import numpy as np
import ase.io
import matplotlib.pyplot as plt
import gradio as gr
from ase.visualize.plot import plot_atoms

from src.algorithm import get_parser, gen_supercell, check_vectors

class StructureHandler:
    def __init__(self, bottom, top, z, d):
        self.bottom = bottom
        self.top = top
        self.z = z
        self.d = d

    def generate_initial_plot(self, angle_start, angle_end, angle_step, tolerance):
        angles = np.arange(angle_start, angle_end, angle_step)
        nums = []
        Deltas = []
        results = []
        valid_angles = []

        for ang in angles:
            result, Delta = gen_supercell(self.bottom, self.top, ang, self.z, self.d)
            if Delta * 100 < tolerance:
                num = len(result)
                nums.append(num)
                Deltas.append(Delta)
                results.append((result, Delta))
                valid_angles.append(ang)

        if nums == []:
            return None, None, None

        nums = np.array(nums)
        Deltas = np.array(Deltas) * 100
        fig, ax = plt.subplots()
        scatter = ax.scatter(valid_angles, nums, c=Deltas, picker=True)
        plt.colorbar(scatter, label='Mismatch (%)')
        plt.xlabel(r'Twist Angle ($^{\circ}$)')
        plt.ylabel('Number of Atoms')
        plt.title('Num vs Angle')
        plt.close()
        return fig, angles, results

    def generate_detailed_structure(self, selected_angle, angles, results):
        try:
            ind = np.isclose(angles, selected_angle, atol=1e-3).tolist().index(True)
            result, Delta = results[ind]
            num = len(result)
            fig, ax = plt.subplots()
            supercell = result.copy()
            supercell.wrap()
            plot_atoms(supercell, ax, scale=2, radii=0.25, rotation=('0x,0y,0z'))
            ax.set_axis_off()
            plt.close()
            return fig
        except ValueError:
            return "Angle not in the generated range. Please enter a valid angle."
    
    def save_structure(self, selected_angle, angles, results, filename):
        try:
            ind = np.isclose(angles, selected_angle, atol=1e-3).tolist().index(True)
            result, _ = results[ind]
            if not filename:
                filename = f"structure_{selected_angle:.2f}.xsf"
            ase.io.write(filename, result)
            return filename
        except ValueError:
            return None

def main_interface(bottom_file, top_file, angle_start, angle_end, angle_step, z, d, tol, selected_angle, save_option, filename):
    try:
        bottom = ase.io.read(bottom_file.name)
        top = ase.io.read(top_file.name)
        check_vectors(bottom)
        check_vectors(top)
        
        handler = StructureHandler(bottom, top, z, d)
        initial_plot, angles, results = handler.generate_initial_plot(angle_start, angle_end, angle_step, tol)
        if initial_plot is None:
            return "No valid structures found in the given range.", None, None, None
        saved_file_path = None
        
        if selected_angle is not None:
            detailed_plot = handler.generate_detailed_structure(selected_angle, angles, results)
            if isinstance(detailed_plot, plt.Figure) and save_option:
                saved_file_path = handler.save_structure(selected_angle, angles, results, filename)
            elif not isinstance(detailed_plot, plt.Figure):
                return detailed_plot, initial_plot, None, None
        else:
            return "Select an angle to view the detailed structure.", initial_plot, None, None
            
        
        if saved_file_path:
            return "Completed! Structure saved to your local directory", initial_plot, detailed_plot, saved_file_path
        return "Completed! Structure not saved", initial_plot, detailed_plot, None
    except SystemExit as e:
        return str(e), None, None, None
    except Exception as e:
        return str(e), None, None, None

iface = gr.Interface(
    fn=main_interface,
    inputs=[
        gr.File(label="Upload bottom layer file"),
        gr.File(label="Upload top layer file"),
        gr.Number(label="Angle start"),
        gr.Number(label="Angle end"),
        gr.Number(label="Angle step"),
        gr.Number(label="z, superlattice constant along z direction"),
        gr.Number(label="d, distance of two layers"),
        gr.Number(label="Maximum mismatch (%)"),
        gr.Number(label="Selected one angle for visualization (Optional)"),
        gr.Checkbox(label="Save Structure", value=False),
        gr.Textbox(label="Filename to save structure, default=structure_angle.xsf")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Plot(label="Initial Scatter Plot"),
        gr.Plot(label="Selected Structure View"),
        gr.File(label="Download Structure")
    ],
    title="Twisted vdWHS Visualization",
    description="Generate and visualize twisted van der Waals heterostructures."
)

iface.launch(inbrowser=True)

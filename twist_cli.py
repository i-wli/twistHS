import numpy as np
import ase.io
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import PathCollection
from matplotlib.widgets import Button

from src.algorithm import get_parser, gen_supercell, check_vectors

class StructureHandler:
    def __init__(self, angles, bottom, top, args, results):
        self.current_structure = None
        self.ang = 0
        self.num = 0
        self.Delta = 0
        self.angles = angles
        self.bottom = bottom
        self.top = top
        self.args = args
        self.results = results

    def onpick(self, event):
        if isinstance(event.artist, PathCollection):
            ind = event.ind[0]
            self.ang = self.angles[ind]
            self.current_result, self.Delta = self.results[ind]
            self.num = len(self.current_result)
            fig, ax = plt.subplots()
            supercell = self.current_result.copy()
            supercell.wrap()
            plot_atoms(supercell, ax, scale=2, radii=0.3, rotation=('0x,0y,0z'))
            plt.title(f"Angle: {self.ang:.2f}, Num: {self.num}, Mismatch: {self.Delta*100:.2f}%")
            ax.set_axis_off()
            axsave = plt.axes([0.81, 0.05, 0.1, 0.075])
            btn_save = Button(axsave, 'Save')
            btn_save.on_clicked(self.save_structure)
            plt.show()

    def save_structure(self, event):
        if self.current_result is not None:
            filename = self.get_filename()
            self.current_result.write(filename)
            print(f"Structure saved as {filename}")
        else:
            print("No structure to save")

    def get_filename(self):
        if self.args.write is None:
            return f"Super_{self.ang:.2f}_{self.num}.xsf"
        else:
            return self.args.write

def main():
    parser = get_parser()
    args = parser.parse_args()
    bottom = ase.io.read(args.bottom)
    top = ase.io.read(args.top)
    check_vectors(bottom)
    check_vectors(top)

    nums = []
    angles = []
    Deltas = []
    results = []

    if args.alist is None:
        angles = args.angle
    else:
        angles = np.arange(args.alist[0], args.alist[1], args.alist[2])

    for ang in angles:
        result, Delta = gen_supercell(bottom, top, ang, args.z, args.d)
        num = len(result)
        nums.append(num)
        Deltas.append(Delta)
        results.append((result, Delta))

    nums = np.array(nums)
    Deltas = np.array(Deltas) * 100
    scatter = plt.scatter(angles, nums, c=Deltas, picker=True)
    plt.colorbar(scatter, label='Mismatch (%)')
    plt.xlabel(r'Twist Angle ($^{\circ}$)')
    plt.ylabel('Number of Atoms')
    plt.title('Click one point to see the structure.')

    handler = StructureHandler(angles, bottom, top, args, results)

    plt.gcf().canvas.mpl_connect('pick_event', handler.onpick)

    plt.show()

if __name__ == '__main__':
    main()
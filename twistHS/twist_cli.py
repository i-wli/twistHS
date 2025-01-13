import numpy as np
import ase.io
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.widgets import Button

from twistHS.lib.algorithm import get_parser, HSGenerator

class StructureVisualizer:
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
            theta_eff = self.current_result.info["theta_eff"]
            self.num = len(self.current_result)
            fig, ax = plt.subplots()
            supercell = self.current_result.copy()
            supercell.wrap()
            plot_atoms(supercell, ax, scale=2, radii=0.3, rotation=('0x,0y,0z'))
            plt.title(f"Effective angle: {theta_eff:.4f}°, Num: {self.num}, Max strain: {self.Delta*100:.2f}%")
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
    if args.angle is None and args.alist is None:
        parser.error('specify at least one of -a/--angle or -al/--alist')
    bottom = ase.io.read(args.bottom)
    top = ase.io.read(args.top)

    twistSupercell = HSGenerator(bottom, top, method=args.method, tolerance=args.tolerance,
                                angle_tol=args.angle_tol, max_atoms=args.max_atoms)

    if args.alist is None:
        angles = args.angle
    else:
        angles = np.arange(args.alist[0], args.alist[1], args.alist[2])

    if args.max_order is not None:
        if args.max_order < 1:
            parser.error('--max-order must be at least 1')
        for ang in angles:
            print(f"Moiré components at {ang:g}° (period in Å, q in Å⁻¹; no Fourier amplitudes)")
            print('order  n  m  q  period  angle (deg)  multiplicity')
            for c in twistSupercell.moire_components(ang, args.max_order):
                print(f'{c.order:.8g}  {c.n}  {c.m}  '
                      f'({c.q[0]:.8g}, {c.q[1]:.8g})  {c.period:.8g}  '
                      f'{c.angle_deg:.8g}  {c.multiplicity}')

    entries = twistSupercell.scan(angles, args.z, args.d)
    if not entries:
        raise SystemExit('[err] No structure within the configured limits')
    angles = np.array([angle for angle, _, _ in entries])
    effective_angles = np.array([result.info["theta_eff"] for _, result, _ in entries])
    nums = np.array([len(result) for _, result, _ in entries])
    Deltas = np.array([strain for _, _, strain in entries]) * 100
    results = [(result, strain) for _, result, strain in entries]
    scatter = plt.scatter(effective_angles, nums, c=Deltas, picker=True)
    plt.colorbar(scatter, label='Max strain (%)')
    plt.xlabel(r'Effective Twist Angle ($^{\circ}$)')
    plt.ylabel('Number of Atoms')
    plt.title('Click one point to see the structure.')

    handler = StructureVisualizer(angles, bottom, top, args, results)

    plt.gcf().canvas.mpl_connect('pick_event', handler.onpick)

    plt.show()

if __name__ == '__main__':
    main()

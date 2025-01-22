import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from twistHS.lib.algorithm import get_parser
from twistHS.lib.workflow import (
    INPUT_ERRORS,
    MOIRE_HEADERS,
    NO_SUPERCELLS,
    angle_range,
    export_structure,
    plot_scan,
    plot_structure,
    run_scan,
)


class StructureVisualizer:
    def __init__(self, entries, filename, outformat):
        self.entries = entries
        self.filename = filename
        self.outformat = outformat
        figure, ax = plt.subplots()
        self.scatter = plot_scan(entries, ax)
        figure.canvas.mpl_connect("pick_event", lambda event: self.onpick(event))

    def onpick(self, event):
        if event.artist is not self.scatter or not len(event.ind):
            return
        indices = sorted(event.ind)
        figure, ax = plt.subplots()
        figure.subplots_adjust(bottom=0.17)
        save = Button(figure.add_axes([0.68, 0.04, 0.27, 0.08]), "Export supercell")
        widgets = [save]
        selected = indices[0]

        def update(index):
            nonlocal selected
            selected = index
            ax.clear()
            plot_structure(self.entries[selected], ax)
            figure.canvas.draw_idle()

        def save_selected(_event):
            try:
                path = export_structure(self.entries[selected], self.filename, self.outformat)
            except INPUT_ERRORS as error:
                print(f"Supercell export failed. {error}")
            else:
                print(f"Exported supercell to {path}")

        if len(indices) > 1:
            figure.subplots_adjust(left=0.4)
            choice_ax = figure.add_axes([0.02, 0.21, 0.33, 0.64], title="Overlapping supercells")
            labels = [f"{index + 1}. Requested twist {self.entries[index].requested_angle:g}°"
                      for index in indices]
            choices = RadioButtons(choice_ax, labels)
            choices.on_clicked(lambda label: update(indices[labels.index(label)]))
            widgets.append(choices)

        save.on_clicked(save_selected)
        figure.canvas.mpl_connect("close_event", lambda _event: widgets.clear())
        update(selected)
        figure.show()


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.outformat and not args.write:
        parser.error("--outformat requires --write")
    try:
        angles = args.angle if args.alist is None else angle_range(*args.alist)
        entries, rows = run_scan(
            args.bottom, args.top, angles, args.z, args.d,
            method=args.method, tolerance=args.tolerance, angle_tol=args.angle_tol,
            moire_tol=args.moire_tol, moire_order=args.moire_order, max_atoms=args.max_atoms,
            backend=args.backend, max_order=args.max_order,
        )
    except INPUT_ERRORS as error:
        parser.error(str(error))

    if args.max_order is not None:
        print("Geometric moiré components")
        print("\t".join(MOIRE_HEADERS))
        for row in rows:
            print("\t".join(row))
        if not rows:
            print("No reciprocal-lattice vector pairs satisfy the analysis cutoff.")
    if not entries:
        raise SystemExit(NO_SUPERCELLS)

    StructureVisualizer(entries, args.write, args.outformat)
    plt.show()


if __name__ == "__main__":
    main()

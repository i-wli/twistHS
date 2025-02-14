from tempfile import TemporaryDirectory, mkdtemp

import gradio as gr
from matplotlib.figure import Figure

from twistHS.lib.algorithm import (
    CONSTRUCTION_METHODS,
    DEFAULT_DELTA_Z,
    DEFAULT_SUPER_Z,
    DESCRIPTION,
)
from twistHS.lib.coincidence import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_MAX_ATOMS,
    DEFAULT_MOIRE_ORDER,
    DEFAULT_MOIRE_TOL,
    DEFAULT_TOLERANCE,
    SEARCH_BACKENDS,
)
from twistHS.lib.workflow import (
    INPUT_ERRORS,
    MOIRE_HEADERS,
    NO_SUPERCELLS,
    STRAIN_LABEL,
    angle_range,
    export_structure,
    plot_scan,
    plot_structure,
    positive_integer,
    run_scan,
)


def scan_bilayer(bottom, top, start, stop, step, z, d, tolerance, method,
                 angle_tol, moire_tol, moire_order, max_atoms, backend, max_order):
    if bottom is None or top is None:
        raise ValueError("Supply the lower- and upper-layer primitive cells.")
    if None in (tolerance, angle_tol, moire_tol, moire_order):
        raise ValueError("Specify the maximum strain, maximum twist-angle deviation, maximum relative change in "
                         "moiré wave vectors, and reciprocal cutoff for the moiré constraint.")
    return run_scan(
        bottom, top, angle_range(start, stop, step), z, d,
        method=method, tolerance=tolerance, angle_tol=angle_tol,
        moire_tol=moire_tol, moire_order=moire_order,
        max_atoms=positive_integer(max_atoms, "The maximum number of atoms per supercell"),
        backend=backend, max_order=max_order,
    )


def preview_supercell(index, entries):
    if index is None:
        return None, None, ""
    figure = Figure(layout="constrained")
    plot_structure(entries[index], figure.subplots())
    return figure, None, ""


def build_interface(download_directory):
    def scan(*inputs):
        try:
            *settings, cutoff = inputs
            try:
                max_order = int(cutoff) if cutoff.strip() else None
            except ValueError as error:
                raise ValueError("The reciprocal cutoff for moiré component analysis must be a positive integer.") from error
            entries, rows = scan_bilayer(*settings, max_order)
        except INPUT_ERRORS as error:
            return str(error), None, gr.update(value=[], visible=False), [], gr.update(choices=[], value=None), None, None, ""
        figure = None
        status = NO_SUPERCELLS
        if entries:
            figure = Figure(layout="constrained")
            plot_scan(entries, figure.subplots())
            status = f"Commensurate supercells found: {len(entries)}."
        if max_order is not None and not rows:
            status += " No reciprocal-lattice vector pairs satisfy the analysis cutoff."
        choices = [(f"{i + 1}. {entry.label}", str(i)) for i, entry in enumerate(entries)]
        return (status, figure, gr.update(value=rows, visible=max_order is not None), entries,
                gr.update(choices=choices, value=None), None, None, "")

    def export(index, entries, filename, outformat):
        if index is None:
            return None, "Select a commensurate supercell to export."
        try:
            directory = mkdtemp(dir=download_directory)
            path = export_structure(entries[index], filename.strip(), outformat.strip(), directory=directory)
        except INPUT_ERRORS as error:
            return None, f"Supercell export failed. {error}"
        return str(path), f"Prepared {path.name} for download."

    with gr.Blocks(title="twistHS · Commensurate moiré supercells") as interface:
        gr.Markdown(f"# twistHS\n{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                with gr.Row():
                    bottom = gr.File(label="Lower-layer primitive cell", type="filepath")
                    top = gr.File(label="Upper-layer primitive cell", type="filepath")
                with gr.Row():
                    start = gr.Number(label="Start twist angle (°)")
                    stop = gr.Number(label="Stop twist angle (°)", info="The stop angle is excluded.")
                    step = gr.Number(label="Twist-angle step (°)")
                with gr.Row():
                    d = gr.Number(DEFAULT_DELTA_Z, label="Interlayer distance (Å)",
                                  info="From the lower- to the upper-layer midplane.")
                    z = gr.Number(DEFAULT_SUPER_Z, label="Out-of-plane supercell lattice parameter (Å)")
                tolerance = gr.Number(DEFAULT_TOLERANCE, label=STRAIN_LABEL,
                                      info="Largest absolute principal strain in either layer.")
                with gr.Accordion("Construction and reciprocal-space analysis", open=False):
                    method = gr.Dropdown(CONSTRUCTION_METHODS, value="search", label="Supercell construction method",
                                         info="Coincidence-lattice search or first-order moiré-cell rounding.")
                    backend = gr.Dropdown(SEARCH_BACKENDS, value="numpy", label="Coincidence-lattice search backend")
                    angle_tol = gr.Number(DEFAULT_ANGLE_TOL, label="Maximum twist-angle deviation (°)",
                                          info="Deviation between requested and effective twist angles in "
                                               "coincidence-lattice search.")
                    max_atoms = gr.Number(DEFAULT_MAX_ATOMS, label="Maximum number of atoms per supercell",
                                          info="Applies to coincidence-lattice search.")
                    moire_tol = gr.Number(DEFAULT_MOIRE_TOL, label="Maximum relative change in moiré wave vectors (%)",
                                          info="Applies to first-order combinations and enabled higher-index wave vectors.")
                    moire_order = gr.Number(DEFAULT_MOIRE_ORDER, label="Reciprocal cutoff for the moiré constraint",
                                            info="In multiples of the length of each layer’s shortest nonzero "
                                                 "reciprocal-lattice vector. Use 1 for the first-order constraint.")
                    max_order = gr.Textbox(label="Reciprocal cutoff for geometric moiré component analysis",
                                           placeholder="Leave blank to omit analysis",
                                           info="Positive integer in multiples of the length of each layer’s shortest "
                                                "nonzero reciprocal-lattice vector.")
                submit = gr.Button("Scan twist angles", variant="primary")
                status = gr.Textbox(label="Scan status", interactive=False)
            with gr.Column(scale=2, min_width=440):
                with gr.Tabs() as results:
                    with gr.Tab("Scan results", id="scan"):
                        scatter = gr.Plot(label="Commensurate supercells by effective twist angle")
                        analysis = gr.Dataframe(headers=list(MOIRE_HEADERS), datatype="str",
                                                label="Geometric moiré components", interactive=False, visible=False)
                    with gr.Tab("Supercell"):
                        selection = gr.Dropdown(choices=[], type="index", label="Commensurate supercell",
                                                interactive=True)
                        with gr.Row():
                            with gr.Column(scale=5, min_width=380):
                                structure = gr.Plot(label="Atomic structure")
                            with gr.Column(scale=2, min_width=220):
                                with gr.Row():
                                    filename = gr.Textbox(label="Download filename",
                                                          placeholder="structure_<requested angle>.xsf")
                                    outformat = gr.Textbox(label="ASE output format",
                                                           info="Optional. An explicit format requires a filename.")
                                download = gr.Button("Prepare supercell download")
                                export_status = gr.Textbox(label="Export status", interactive=False)
                                file = gr.File(label="Supercell download", interactive=False)
        entries = gr.State([])

        submit.click(lambda: gr.Tabs(selected="scan"), None, results).then(
            scan,
            [bottom, top, start, stop, step, z, d, tolerance, method, angle_tol, moire_tol, moire_order,
             max_atoms, backend, max_order],
            [status, scatter, analysis, entries, selection, structure, file, export_status],
        )
        selection.input(preview_supercell, [selection, entries], [structure, file, export_status])
        download.click(export, [selection, entries, filename, outformat], [file, export_status])
    return interface


def main():
    with TemporaryDirectory(prefix="twisths-") as download_directory:
        build_interface(download_directory).launch(inbrowser=True)


if __name__ == "__main__":
    main()

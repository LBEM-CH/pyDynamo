"""dcp GUI (Tk): the windows of Dynamo 1.1.514's dcp (dynamo_project_manager_easy and its wizards), drawn in Tk.

  dcp                 (shell)  open the project manager in the current folder
  dcp <project>       (shell)  open it on ./<project> (Dynamo's dcp(name): a missing project is only reported)
  dcp.open('p')       (Python) the same, from pydyn import dcp

All logic is in pydyn.dcpcontrol.DcpController (pure Python, tested without a display). This module only draws
widgets, forwards input and shows the controller's messages. Window sequence (S/@dcp/stepOrdering.m):
project -> data -> table -> template -> masks -> numerical parameters (rounds) -> computing environment -> check ->
unfold; the buttons are enabled step by step as in Dynamo (stepStatus2GUI / enableCascade). Parameters are written
to disk as soon as they are entered, as in Dynamo (no separate save).

The masks window has the three mask-editor buttons (dynamo_mask_GUI: cylinder masks, pydyn.masks; the file appears
in the mask box and enters the project with Enter / 'enter'). The menu "Multireference" holds "Create an even/odd
project" and 'Adaptive filtering ("golden standard")' > "Derive a project." / "Edit parameters for Adaptive Filtering
run" (the 7-parameter editor; each edit is saved at once).

Needs an X display (DISPLAY). Without one, main() and open_project() print a clear message and return.
"""
import os
import queue
import sys
import threading
import traceback

PYDYN_TITLE = "pydyn dcp (Dynamo 1.1.514 project manager)"
_COLORS = {"k": "black", "r": "red3", "warning": "dark orange", "waiting": "blue", "b": "blue", "g": "dark green",
           "none": "black"}


def display_problem():
    """None if Tk can probably open a window, else the reason (no DISPLAY / no tkinter)."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return ("no X display: the DISPLAY variable is not set. Run 'dcp' from a desktop session (or ssh -X / "
                "a VNC session); everything the GUI does is also available from Python: "
                "from pydyn import dcp, dvput, dvcheck, dvunfold")
    try:
        import tkinter  # noqa: F401
    except Exception as e:                           # pragma: no cover - depends on the Python build
        return "tkinter is not available in this Python (%s)" % e
    return None


def _slice_image(tk, vol, axis, mask=None, zoom=3):
    """Central slice of the template (grey) with the mask region tinted red, as a Tk PhotoImage (PPM)."""
    import numpy as np
    n = vol.shape[axis]
    s = np.take(vol, n // 2, axis=axis).astype(np.float64)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    g = np.clip((s - lo) / (hi - lo + 1e-12), 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    if mask is not None:
        m = np.take(mask, n // 2, axis=axis)
        a = np.clip(m, 0, 1) * 0.45
        rgb[..., 0] = rgb[..., 0] * (1 - a) + a
        rgb[..., 1] *= (1 - a)
        rgb[..., 2] *= (1 - a)
    rgb = np.kron((rgb * 255).astype(np.uint8), np.ones((zoom, zoom, 1), dtype=np.uint8))
    h, w = rgb.shape[:2]
    return tk.PhotoImage(data=b"P6 %d %d 255\n" % (w, h) + rgb.tobytes(), format="PPM")


class DcpApp:
    """The main window (dynamo_project_manager_easy) and its wizards."""

    def __init__(self, root, project=None):
        import tkinter as tk
        from tkinter import ttk
        from .dcpcontrol import DcpController
        self.tk, self.ttk, self.root = tk, ttk, root
        self._pending = queue.Queue()                 # messages from the worker thread ("average all")
        self.c = DcpController(echo=self._echo)
        self.windows = {}
        root.title(PYDYN_TITLE)
        f = ttk.Frame(root, padding=8)
        f.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        ttk.Label(f, text="project").grid(row=0, column=0, sticky="w")
        self.v_project = tk.StringVar(value=project or "")
        e = ttk.Entry(f, textvariable=self.v_project, width=30)
        e.grid(row=0, column=1, sticky="ew")
        e.bind("<Return>", lambda ev: self.enter_project())
        ttk.Button(f, text="open / create", command=self.enter_project).grid(row=0, column=2, padx=2)
        ttk.Button(f, text="projects here...", command=self.choose_project).grid(row=0, column=3, padx=2)
        ttk.Label(f, text="references").grid(row=0, column=4, sticky="e", padx=(12, 2))
        self.v_nref = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.v_nref, width=3).grid(row=0, column=5, sticky="w")
        steps = ttk.Frame(f)
        steps.grid(row=1, column=0, columnspan=6, sticky="w", pady=6)
        self.buttons = {}
        for i, (step, label, cmd) in enumerate((
                ("data", "data", self.win_data), ("table", "table", self.win_table),
                ("template", "template", self.win_template), ("masks", "masks", self.win_masks),
                ("settings", "numerical parameters", self.win_rounds),
                ("destination", "computing environment", self.win_environment),
                ("check", "check", self.do_check), ("unfold", "unfold", self.do_unfold),
                ("run", "run...", self.do_run))):
            b = ttk.Button(steps, text=label, command=cmd)
            b.grid(row=0, column=i, padx=2)
            self.buttons[step] = b
        self.log = tk.Text(f, height=16, width=110, wrap="word")
        self.log.grid(row=2, column=0, columnspan=6, sticky="nsew")
        for k, col in _COLORS.items():
            self.log.tag_configure(k, foreground=col)
        f.rowconfigure(2, weight=1)
        f.columnconfigure(1, weight=1)
        self._build_menus()
        self._echo("Folder: %s" % os.getcwd(), "k")
        self._refresh_buttons()
        if project:
            self.enter_project()

    def _build_menus(self):
        """Menu "Multireference" of dynamo_project_manager_easy.fig (labels as in the .fig, 10b 4.1)."""
        tk = self.tk
        bar = tk.Menu(self.root)
        mr = tk.Menu(bar, tearoff=0)
        mr.add_command(label="Create an even/odd project",
                       command=lambda: self._guard(self.c.create_even_odd_project))
        gs = tk.Menu(mr, tearoff=0)
        gs.add_command(label="Derive a project.", command=lambda: self._guard(self.c.golden_standard_derive))
        gs.add_command(label="Edit parameters for Adaptive Filtering run", command=self.win_abp)
        gs.add_command(label="Plot attained resolution FSC for all iterations",
                       command=lambda: self._guard(self.c.plot_adaptive_filtering_fsc))
        mr.add_cascade(label='Adaptive filtering ("golden standard")', menu=gs)
        # menu "View" (uimenu_results of dynamo_project_manager_easy.fig; callbacks :1099-1105); the viewers are
        # separate Qt processes (pydyn.viewers)
        vw = tk.Menu(bar, tearoff=0)
        vw.add_command(label="Last computed average(s)", command=lambda: self._guard(self.c.view_last_averages))
        vw.add_command(label="Last computed table [not ported in pydyn]", state="disabled")
        vw.add_command(label="All seed files [not ported in pydyn]", state="disabled")
        bar.add_cascade(label="View", menu=vw)
        bar.add_cascade(label="Multireference", menu=mr)
        self.root.config(menu=bar)
        self.menus = {"bar": bar, "multireference": mr, "golden_standard": gs, "view": vw}

    # ------------------------------------------------------------------------------------------------ plumbing
    def _echo(self, text, color="k"):
        if threading.current_thread() is not threading.main_thread():
            self._pending.put((text, color))         # Tk is used from the main thread only
            return
        self._flush()
        self.log.insert("end", text + "\n", color if color in _COLORS else "k")
        self.log.see("end")

    def _flush(self):
        while True:
            try:
                text, color = self._pending.get_nowait()
            except queue.Empty:
                return
            self.log.insert("end", text + "\n", color if color in _COLORS else "k")
            self.log.see("end")

    def _guard(self, fn, *a, **kw):
        """Run a controller action; Dynamo errors and unported paths become red messages, not crashes."""
        from .vpr import DynamoError
        try:
            r = fn(*a, **kw)
        except NotImplementedError as e:
            self._echo("[not ported in pydyn] %s" % e, "r")
            r = None
        except DynamoError as e:
            self._echo(str(e), "r")
            r = None
        except Exception as e:                       # show, do not crash the GUI
            self._echo("ERROR: %s" % e, "r")
            self._echo(traceback.format_exc(), "r")
            r = None
        self._refresh_buttons()
        return r

    def _refresh_buttons(self):
        en = self.c.enabled
        has = self.c.name is not None
        for step, b in self.buttons.items():
            on = has if step == "check" else (has and en.get(step, False))
            b.state(["!disabled"] if on else ["disabled"])
        nref = ""
        if has and self.c.runtime is not None:
            try:
                nref = "%d" % int(float(self.c.runtime["nref_r1"]))
            except (KeyError, TypeError, ValueError):
                nref = "?"
        self.v_nref.set(nref)

    def _toplevel(self, key, title):
        tk = self.tk
        w = self.windows.get(key)
        if w is not None and w.winfo_exists():
            w.destroy()
        w = tk.Toplevel(self.root)
        w.title('%s for project "%s"' % (title, self.c.name))
        self.windows[key] = w
        return w

    def _file_row(self, parent, row, label, value, enter, kind="file", width=60):
        tk, ttk = self.tk, self.ttk
        from tkinter import filedialog
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        v = tk.StringVar(value=value)
        e = ttk.Entry(parent, textvariable=v, width=width)
        e.grid(row=row, column=1, sticky="ew")

        def do_enter(*_):
            r = self._guard(enter, v.get().strip())
            e.configure(foreground="black" if r is not None and getattr(r, "ok", False) else "red3")
            return r

        def browse():
            p = filedialog.askdirectory(initialdir=os.getcwd()) if kind == "dir" else \
                filedialog.askopenfilename(initialdir=os.getcwd())
            if p:
                rel = os.path.relpath(p)
                v.set(p if rel.startswith("..") else rel)
                do_enter()
        e.bind("<Return>", do_enter)
        ttk.Button(parent, text="browse", command=browse).grid(row=row, column=2, padx=2)
        ttk.Button(parent, text="enter", command=do_enter).grid(row=row, column=3, padx=2)
        return v, e

    # ------------------------------------------------------------------------------------------------ project
    def enter_project(self):
        from tkinter import messagebox
        name = self.v_project.get().strip()
        st = self._guard(self.c.enter_project_name, name)
        if st == "not_a_project":
            if messagebox.askyesno("Not a project",
                                   'The string "%s" is not a project currently in file system.\n\n'
                                   "Create a new project? (dcp.create: a blank project)" % name):
                self._guard(self.c.create, name)
            else:
                self._echo("Ok. No changes in file system.")
        self._refresh_buttons()

    def choose_project(self):
        tk, ttk = self.tk, self.ttk
        projects = self.c.list_projects()
        if not projects:
            self._echo("No projects in %s" % os.getcwd(), "warning")
            return
        w = tk.Toplevel(self.root)
        w.title("projects in %s" % os.getcwd())
        lb = tk.Listbox(w, height=min(15, len(projects)), width=40)
        for p in projects:
            lb.insert("end", p)
        lb.grid(row=0, column=0, padx=6, pady=6)

        def pick(*_):
            sel = lb.curselection()
            if sel:
                self.v_project.set(projects[sel[0]])
                w.destroy()
                self.enter_project()
        lb.bind("<Double-Button-1>", pick)
        ttk.Button(w, text="open", command=pick).grid(row=1, column=0, pady=4)

    # ------------------------------------------------------------------------------------------------ data
    def win_data(self):
        ttk = self.ttk
        w = self._toplevel("data", "Input a data folder")
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        facts = self.tk.StringVar(value="")

        def enter(folder):
            r = self.c.data_enter(folder)
            df = self.c.data_facts
            if r.ok and df:
                facts.set("%d particles of size %d (padding %s)" % (df["N"], df["l"], df["padding"]))
            return r
        self._file_row(f, 0, "data folder", self.c.runtime["folder_data"], enter, kind="dir")
        ttk.Label(f, textvariable=facts).grid(row=1, column=1, sticky="w")
        ttk.Button(f, text="ok", command=w.destroy).grid(row=2, column=3, pady=4)

    # ------------------------------------------------------------------------------------------------ table
    def win_table(self):
        tk, ttk = self.tk, self.ttk
        w = self._toplevel("table", "Input initial table(s)")
        insp = tk.Menu(w, tearoff=0)                 # dynamo__wizard_table.m uimenu_inspect
        insp.add_command(label="General info [not ported in pydyn]", state="disabled")
        insp.add_command(label="Plot orientations", command=lambda: self._guard(self.c.plot_orientations))
        mb = tk.Menu(w)
        mb.add_cascade(label="Inspect", menu=insp)
        w.config(menu=mb)
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        self._file_row(f, 0, "table", self.c.runtime["file_table_initial"], self.c.table_enter)
        b = ttk.Frame(f)
        b.grid(row=1, column=1, sticky="w", pady=4)
        ttk.Button(b, text="blank table", command=lambda: self._guard(self.c.table_blank)).grid(row=0, column=0)
        ttk.Button(b, text="random table", command=lambda: self._guard(self.c.table_random)).grid(row=0, column=1)
        ttk.Button(f, text="ok", command=w.destroy).grid(row=2, column=3, pady=4)

    # ------------------------------------------------------------------------------------------------ template
    def win_template(self):
        tk, ttk = self.tk, self.ttk
        w = self._toplevel("template", "Input initial template(s)")
        insp = tk.Menu(w, tearoff=0)                 # dynamo__wizard_template.m uimenu_inspect > uimenu_dview
        insp.add_command(label="dview (gray scale browser)", command=lambda: self._guard(self.c.view_template))
        mb = tk.Menu(w)
        mb.add_cascade(label="Inspect", menu=insp)
        w.config(menu=mb)
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        v, _ = self._file_row(f, 0, "template", self.c.runtime["file_template_initial"], self.c.template_enter)
        box = ttk.LabelFrame(f, text="average all particles (dynamo_average of the data with the project table)",
                             padding=6)
        box.grid(row=1, column=0, columnspan=4, sticky="ew", pady=6)
        o = {k: tk.BooleanVar(value=False) for k in ("random", "random_table", "fcompensate")}
        ttk.Checkbutton(box, text="random sample", variable=o["random"]).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(box, text="random table", variable=o["random_table"]).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(box, text="Fourier compensation", variable=o["fcompensate"]).grid(row=0, column=2, sticky="w")
        status = tk.StringVar(value="")
        ttk.Label(box, textvariable=status).grid(row=1, column=0, columnspan=3, sticky="w")

        def average():
            opts = dict(random_sample=o["random"].get(), random_table=o["random_table"].get(),
                        fcompensate=o["fcompensate"].get())
            if any(opts.values()):
                self._guard(self.c.template_average, **opts)
                return
            status.set("averaging (GPU if available)... the window stays usable")
            result = {}

            def work():
                try:
                    result["r"] = self.c.template_average()
                except BaseException as e:           # reported in the Tk thread
                    result["e"] = e

            th = threading.Thread(target=work, daemon=True)
            th.start()

            def done():
                self._flush()
                if th.is_alive():
                    self.root.after(500, done)
                    return
                if "e" in result:
                    self._echo("average all failed: %s" % result["e"], "r")
                else:
                    v.set(self.c.runtime["file_template_initial"])
                status.set("")
                self._refresh_buttons()
            self.root.after(500, done)
        ttk.Button(box, text="average all", command=average).grid(row=0, column=3, padx=6)
        ttk.Button(f, text="ok", command=w.destroy).grid(row=2, column=3, pady=4)

    # ------------------------------------------------------------------------------------------------ masks
    def win_masks(self):
        tk, ttk = self.tk, self.ttk
        w = self._toplevel("masks", "Input masks")
        mb = tk.Menu(w)                              # dynamo__wizard_masks.m: "View mask" and "View Classification mask"
        for label, mkind in (("View mask", "mask"), ("View Classification mask", "cmask")):
            vm = tk.Menu(mb, tearoff=0)
            vm.add_command(label="dview", command=lambda m=mkind: self._guard(self.c.view_mask, m))
            for ax in ("x", "y", "z"):
                vm.add_command(label="overlay %s" % ax,
                               command=lambda a=ax, m=mkind: self._guard(self.c.overlay_mask, a, m))
            mb.add_cascade(label=label, menu=vm)
        w.config(menu=mb)
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        rt = self.c.runtime
        rows = {}
        self.mask_rows = {}                          # (StringVar, Entry) per box
        for i, (k, label, field, enter) in enumerate((
                ("mask", "alignment mask", "file_mask", self.c.mask_enter),
                ("cmask", "classification mask", "file_mask_classification", self.c.cmask_enter),
                ("fmask", "Fourier mask (fmask)", "file_fmask_initial", self.c.fmask_enter),
                ("smask", "smoothing mask (FSC)", "file_mask_smoothing", self.c.smask_enter))):
            v_e = self._file_row(f, i, label, rt[field], enter)
            rows[k] = v_e[0]
            self.mask_rows[k] = v_e
        size = self.c.template_size_cache
        ttk.Label(f, text="template size: %s" % ("<?>" if size is None else " ".join(map(str, size)))).grid(
            row=4, column=1, sticky="w")
        semi = tk.StringVar(value=self.c.masks_window_opened() or "")
        g = ttk.LabelFrame(f, text="generate", padding=6)
        g.grid(row=5, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Label(g, text="semiaxes (x y z)").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=semi, width=14).grid(row=0, column=1, sticky="w")

        def refresh():
            for k, field in (("mask", "file_mask"), ("cmask", "file_mask_classification"),
                             ("fmask", "file_fmask_initial"), ("smask", "file_mask_smoothing")):
                rows[k].set(self.c.runtime[field])
            semi.set(self.c.semiaxis)
            show()

        def act(fn, *a):
            self._guard(fn, *a)
            refresh()
        ttk.Button(g, text="ellipsoid -> alignment mask",
                   command=lambda: act(self.c.masks_ellipsoid, semi.get())).grid(row=0, column=2, padx=2)
        ttk.Button(g, text="copy mask -> classification mask",
                   command=lambda: act(self.c.masks_copy_mask)).grid(row=0, column=3, padx=2)
        ttk.Button(g, text="fmask: use ones", command=lambda: act(self.c.masks_use_ones)).grid(row=1, column=2, padx=2)
        ttk.Button(g, text="smask: use ones", command=lambda: act(self.c.smask_use_ones)).grid(row=1, column=3, padx=2)
        ttk.Button(g, text="use defaults", command=lambda: act(self.c.masks_use_defaults)).grid(row=1, column=0,
                                                                                               columnspan=2, sticky="w")
        def editor(kind):
            r = self._guard(self.c.mask_editor, kind)
            if r:
                ed, box, file = r
                rows[box].set(file)                  # the box shows the file in blue: press Enter to accept it
                self.win_mask_editor(ed, on_saved=show)
        for i, (kind, label) in enumerate((("align", "mask editor"), ("cmask", "classification mask editor"),
                                           ("smask", "smoothing mask editor"))):
            ttk.Button(g, text=label, command=lambda k=kind: editor(k)).grid(row=2, column=i, sticky="w", padx=2)
        pv = ttk.LabelFrame(f, text="overlay on the template (central slices: z, y, x)", padding=6)
        pv.grid(row=6, column=0, columnspan=4, sticky="ew")
        which = tk.StringVar(value="mask")
        for i, k in enumerate(("mask", "cmask", "smask", "fmask")):
            ttk.Radiobutton(pv, text=k, value=k, variable=which, command=lambda: show()).grid(row=0, column=i,
                                                                                               sticky="w")
        labels = [ttk.Label(pv) for _ in range(3)]
        for i, lab in enumerate(labels):
            lab.grid(row=1, column=i, padx=3)
        info = tk.StringVar(value="")
        ttk.Label(pv, textvariable=info).grid(row=2, column=0, columnspan=4, sticky="w")
        keep = {}

        def show():
            try:
                pair = self.c.preview(which.get())
            except Exception as e:
                info.set("no preview: %s" % e)
                return
            if pair is None:
                info.set("no preview (missing file or size mismatch)")
                for lab in labels:
                    lab.configure(image="")
                return
            t, m = pair
            zoom = max(1, min(4, 256 // max(t.shape)))
            keep["imgs"] = [_slice_image(tk, t, ax, m, zoom) for ax in (0, 1, 2)]
            for lab, im in zip(labels, keep["imgs"]):
                lab.configure(image=im)
            info.set("%s: %d voxels > 0, %d distinct values" % (which.get(), int((m > 0).sum()),
                                                                 len(set(m.ravel().tolist()[:200000]))))
        show()
        ttk.Button(f, text="ok", command=lambda: (self._masks_closed(), w.destroy())).grid(row=7, column=3, pady=4)
        w.protocol("WM_DELETE_WINDOW", lambda: (self._masks_closed(), w.destroy()))

    def win_mask_editor(self, ed, on_saved=None):
        """dynamo_mask_GUI as opened by the masks wizard: size, type popup, the radio buttons with their edit
        boxes (all off), invert, output file, save (pydyn.masks.MaskEditor)."""
        tk, ttk = self.tk, self.ttk
        from .masks import GUI_RADIO_SETTINGS, GUI_TYPES
        w = self._toplevel("mask_editor", "Mask editor")
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        v_size = tk.StringVar(value=ed.size)
        ttk.Label(f, text="size").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=v_size, width=16).grid(row=0, column=1, sticky="w")
        v_type = tk.StringVar(value=ed.type)
        ttk.Label(f, text="type").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=v_type, values=GUI_TYPES, state="readonly", width=16).grid(row=1, column=1,
                                                                                            sticky="w")
        radios, edits = {}, {}
        for i, p in enumerate(GUI_RADIO_SETTINGS):
            radios[p] = tk.BooleanVar(value=ed.radio[p])
            edits[p] = tk.StringVar(value=ed.edit[p])
            ttk.Checkbutton(f, text=p, variable=radios[p]).grid(row=2 + i // 2, column=(i % 2) * 2, sticky="w")
            ttk.Entry(f, textvariable=edits[p], width=12).grid(row=2 + i // 2, column=(i % 2) * 2 + 1, sticky="w")
        n = 2 + (len(GUI_RADIO_SETTINGS) + 1) // 2
        v_inv = tk.BooleanVar(value=ed.radio["invert"])
        ttk.Checkbutton(f, text="invert", variable=v_inv).grid(row=n, column=0, sticky="w")
        v_save = tk.StringVar(value=ed.save_file)
        ttk.Label(f, text="save as").grid(row=n + 1, column=0, sticky="w")
        ttk.Entry(f, textvariable=v_save, width=50).grid(row=n + 1, column=1, columnspan=3, sticky="ew")
        self.mask_editor_vars = {"size": v_size, "type": v_type, "radio": radios, "edit": edits, "invert": v_inv,
                                 "save": v_save, "editor": ed}

        def save():
            ed.size = v_size.get()
            ed.set_type(v_type.get())
            for p in GUI_RADIO_SETTINGS:
                ed.select(p, edits[p].get(), on=radios[p].get())
            ed.radio["invert"] = bool(v_inv.get())
            ed.save_file = v_save.get().strip()
            m = self._guard(ed.save)
            for msg in ed.messages:
                self._echo("[mask editor] %s" % msg, "k" if m is not None else "r")
            ed.messages.clear()
            if m is not None and on_saved:
                on_saved()
            return m
        ttk.Button(f, text="save", command=save).grid(row=n + 2, column=3, pady=4)
        ttk.Button(f, text="view", width=5, command=lambda: show_msgs(ed.view_file, v_save.get().strip())).grid(
            row=n + 1, column=4, padx=2)
        # "Show" panel (dynamo_mask_GUI): mask | overlay, x | y | z, alpha, colour, "show" (dslices in a Qt process)
        sp = ttk.LabelFrame(f, text="Show", padding=4)
        sp.grid(row=n + 3, column=0, columnspan=5, sticky="ew", pady=4)
        v_mode, v_view = tk.StringVar(value="mask"), tk.StringVar(value="z")
        v_alpha, v_color = tk.StringVar(value="0.5"), tk.StringVar(value="r")
        for i, (txt, val) in enumerate((("mask", "mask"), ("overlay", "overlay"))):
            ttk.Radiobutton(sp, text=txt, value=val, variable=v_mode).grid(row=0, column=i, sticky="w")
        for i, ax in enumerate(("x", "y", "z")):
            ttk.Radiobutton(sp, text=ax, value=ax, variable=v_view).grid(row=0, column=2 + i, sticky="w")
        ttk.Label(sp, text="alpha").grid(row=1, column=0, sticky="e")
        ttk.Entry(sp, textvariable=v_alpha, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(sp, text="color").grid(row=1, column=2, sticky="e")
        ttk.Entry(sp, textvariable=v_color, width=5).grid(row=1, column=3, sticky="w")

        def show_msgs(fn, *a, **kw):
            r = self._guard(fn, *a, **kw)
            for msg in ed.messages:
                self._echo("[mask editor] %s" % msg, "k")
            ed.messages.clear()
            return r

        def do_show():
            ed.size = v_size.get()
            ed.set_type(v_type.get())
            for p in GUI_RADIO_SETTINGS:
                ed.select(p, edits[p].get(), on=radios[p].get())
            ed.radio["invert"] = bool(v_inv.get())
            try:
                alpha = float(v_alpha.get())
            except ValueError:
                alpha = 0.5
            show_msgs(ed.show, v_mode.get(), v_view.get(), alpha, v_color.get().strip() or "r")
        ttk.Button(sp, text="show", command=do_show).grid(row=1, column=4, padx=6)
        w.mask_editor_show = do_show
        w.mask_editor_save = save
        return w

    def win_abp(self):
        """"Edit parameters for Adaptive Filtering run": the 7 parameters of ABPManager/launchGUI.m; each edit is
        saved at once (dcp.setParameter, a GUI-style save)."""
        tk, ttk = self.tk, self.ttk
        rows = self._guard(self.c.abp_parameters)
        if not rows:
            return None
        w = self._toplevel("abp", "Adaptive bandpass parameters")
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        self.abp_vars = {}
        for i, (name, label, value) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value=value.strip())
            e = ttk.Entry(f, textvariable=v, width=12)
            e.grid(row=i, column=1, sticky="w")

            def commit(_ev=None, name=name, v=v, e=e):
                r = self._guard(self.c.abp_set, name, v.get())
                e.configure(foreground="black" if r is not None and r.ok else "red3")
                return r
            e.bind("<Return>", commit)
            ttk.Button(f, text="enter", command=commit).grid(row=i, column=2, padx=2)
            self.abp_vars[name] = (v, commit)
        ttk.Button(f, text="ok", command=w.destroy).grid(row=len(rows), column=2, pady=4)
        return w

    def _masks_closed(self):
        """dcp.closeRequestMasks: masks step right/wrong from the isCorrect flags."""
        c = self.c
        ok = all(c.correct.get(k) for k in ("mask", "cmask", "fmask"))
        if not ok:
            self._echo("Masks are not yet correct!", "r")
        c._step_status("masks", "right" if ok else "wrong")
        self._refresh_buttons()

    # ------------------------------------------------------------------------------------------------ rounds
    def win_rounds(self):
        tk, ttk = self.tk, self.ttk
        from .dcpcontrol import ROUND_AREAS, ROUND_AREAS_ON
        w = self._toplevel("rounds", "Wizard: Numerical parameters")
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        w.columnconfigure(0, weight=1)
        w.rowconfigure(0, weight=1)
        flt = ttk.Frame(f)
        flt.grid(row=0, column=0, sticky="w")
        areas = {a: tk.BooleanVar(value=a in ROUND_AREAS_ON) for a in ROUND_AREAS}
        for i, a in enumerate(ROUND_AREAS):
            ttk.Checkbutton(flt, text=a, variable=areas[a], command=lambda: fill()).grid(row=0, column=i, sticky="w")
        adv = tk.BooleanVar(value=False)
        ttk.Checkbutton(flt, text="advanced", variable=adv, command=lambda: fill()).grid(row=1, column=0, sticky="w")
        mode = tk.BooleanVar(value=True)                           # True: round parameters

        def toggle():
            mode.set(not mode.get())
            tb.configure(text="show general parameters" if mode.get() else "show round parameters")
            fill()
        tb = ttk.Button(flt, text="show general parameters", command=toggle)
        tb.grid(row=1, column=1, columnspan=2, sticky="w")
        tv = ttk.Treeview(f, show="tree headings", height=22)
        tv.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(f, orient="vertical", command=tv.yview)
        sb.grid(row=1, column=1, sticky="ns")
        tv.configure(yscrollcommand=sb.set)
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)
        comment = tk.StringVar(value="double-click a cell to edit it; Return saves it into the project")
        ttk.Label(f, textvariable=comment, wraplength=900).grid(row=2, column=0, sticky="w", pady=4)
        seq = tk.StringVar(value="")
        ttk.Label(f, textvariable=seq).grid(row=3, column=0, sticky="w")
        rb = ttk.Frame(f)
        rb.grid(row=4, column=0, sticky="w", pady=4)
        state = {"rows": [], "cols": []}

        def fill():
            sel = [a for a in ROUND_AREAS if areas[a].get()]
            rows, cols = self._guard(self.c.rounds_table, sel, adv.get(), mode.get()) or ([], [])
            state["rows"], state["cols"] = rows, cols
            tv.delete(*tv.get_children())
            act = self.c.active_rounds()
            ids = ["c%d" % i for i in range(len(cols))]
            tv.configure(columns=ids)
            tv.heading("#0", text="parameter")
            tv.column("#0", width=260, stretch=False)
            for i, (cid, cn) in enumerate(zip(ids, cols)):
                used = mode.get() and (i + 1) in act
                tv.heading(cid, text=cn + (" *" if used else ""))
                tv.column(cid, width=90 if mode.get() else 300, anchor="w")
            for name, label, cells in rows:
                tv.insert("", "end", iid=name, text=label, values=cells)
            if mode.get():
                its = self.c.iteration_rounds()
                seq.set("rounds with iterations (*): %s;  iteration -> round: %s (%d iterations)"
                        % (act or "none", its, len(its)))
            else:
                seq.set("")

        def on_select(*_):
            s = tv.selection()
            if s:
                comment.set("%s: %s" % (s[0], self.c.comment(s[0])))

        def on_double(ev):
            row = tv.identify_row(ev.y)
            col = tv.identify_column(ev.x)
            if not row or col == "#0":
                return
            j = int(col[1:]) - 1
            x, y, wd, ht = tv.bbox(row, col)
            old = tv.item(row, "values")[j]
            e = ttk.Entry(tv)
            e.insert(0, old.strip())
            e.place(x=x, y=y, width=wd, height=ht)
            e.focus_set()

            done = []

            def commit(*_):                          # Return or clicking elsewhere ends the edit, as in a uitable
                if done:
                    return
                done.append(1)
                txt = e.get()
                e.destroy()
                if txt.strip() == old.strip():
                    return
                rnd = (j + 1) if mode.get() else None
                self._guard(self.c.set_value, row, txt, rnd)
                fill()

            def cancel(*_):
                done.append(1)
                e.destroy()
            e.bind("<Return>", commit)
            e.bind("<KP_Enter>", commit)
            e.bind("<FocusOut>", commit)
            e.bind("<Escape>", cancel)
            self._rounds_cell_commit = commit                  # (used by the smoke test)
        tv.bind("<<TreeviewSelect>>", on_select)
        tv.bind("<Double-Button-1>", on_double)
        self._rounds_cell_editor = on_double                   # (used by the smoke test)

        def add_round():
            self._guard(self.c.add_round)
            fill()

        def remove_round():
            self._guard(self.c.remove_round)
            fill()
        ttk.Button(rb, text="add round (iterations of the next round = 1)", command=add_round).grid(row=0, column=0)
        ttk.Button(rb, text="remove last round (its iterations = 0)", command=remove_round).grid(row=0, column=1,
                                                                                               padx=4)
        ttk.Button(rb, text="refresh", command=fill).grid(row=0, column=2, padx=4)
        ttk.Button(rb, text="ok", command=w.destroy).grid(row=0, column=3, padx=4)
        fill()

    # ------------------------------------------------------------------------------------------------ environment
    def win_environment(self):
        tk, ttk = self.tk, self.ttk
        from .dcpcontrol import DESTINATIONS, ENVIRONMENT_PARAMETERS, PYDYN_EXECUTES, parameter
        w = self._toplevel("environment", "Input execution settings")
        f = ttk.Frame(w, padding=8)
        f.grid(sticky="nsew")
        dest = tk.StringVar(value=self.c.runtime["destination"])
        box = ttk.LabelFrame(f, text="destination (pydyn executes %s)" % " / ".join(PYDYN_EXECUTES), padding=6)
        box.grid(row=0, column=0, columnspan=3, sticky="ew")
        for i, d in enumerate(DESTINATIONS):
            ttk.Radiobutton(box, text=d, value=d, variable=dest,
                            command=lambda d=d: self._guard(self.c.set_destination, d)).grid(row=i // 4,
                                                                                           column=i % 4, sticky="w")
        vals = self.c.environment_values()
        for i, p in enumerate(ENVIRONMENT_PARAMETERS, 1):
            ttk.Label(f, text=parameter(p)["pretty"]).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value=vals[p][0].strip())
            e = ttk.Entry(f, textvariable=v, width=30, foreground="black" if vals[p][1] else "red3")
            e.grid(row=i, column=1, sticky="w")

            def commit(_ev=None, p=p, v=v, e=e):
                r = self._guard(self.c.environment_edit, p, v.get())
                e.configure(foreground="black" if r is not None and r.ok else "red3")
            e.bind("<Return>", commit)
            ttk.Button(f, text="enter", command=commit).grid(row=i, column=2, padx=2)
        ttk.Button(f, text="ok", command=w.destroy).grid(row=len(ENVIRONMENT_PARAMETERS) + 1, column=2, pady=4)

    # ------------------------------------------------------------------------------------------------ check / unfold
    def do_check(self):
        self._guard(self.c.check)

    def do_unfold(self):
        r = self._guard(self.c.unfold)
        if r and r[1] == "ok":
            self._echo("Run it from a terminal in %s:   %s" % (os.getcwd(), self.c.execution_order()), "b")

    def do_run(self):
        from tkinter import messagebox
        order = self._guard(self.c.execution_order)
        if order:
            self._echo("Execution order: %s  (run it from a terminal in %s)" % (order, os.getcwd()), "b")
            messagebox.showinfo("run", "Run this from a terminal in\n%s\n\n%s" % (os.getcwd(), order))


# ====================================================================================================================
# entry points (contract with pydyn.cli / pydyn.dcptool)
# ====================================================================================================================

def _run(project=None):
    problem = display_problem()
    if problem:
        print("[pydyn dcp] cannot open the GUI: %s" % problem)
        return 2
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print("[pydyn dcp] cannot open the GUI: Tk cannot connect to the display %r (%s)"
              % (os.environ.get("DISPLAY"), e))
        return 2
    DcpApp(root, project)
    root.mainloop()
    return 0


def open_project(name):
    """dcp.open(name): the project manager on ./<name> (blocks until the window is closed)."""
    rc = _run(name)
    return None if rc else name


def main(argv=None):
    """The `dcp` command. dcp(name) in Dynamo (S/@dcp/dcp.m:83-140): an existing project is opened; a missing one
    is only reported ("Project NOT in disk ... Creating a new one." and return, without creating it)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if len(argv) > 1:
        print("[pydyn dcp] usage: dcp [project]")
        return 2
    if argv:
        name = argv[0]
        if name == "." or "*" in name:
            print("[pydyn dcp] project search patterns (dynamo_projects 'regexp') are not ported; "
                  "use 'projects here...' in the GUI")
            return 2
        from . import vpr as V
        if not V.isproject(name):
            print('Project NOT in disk: "%s". Creating a new one.' % name)
            print("[pydyn dcp] as in Dynamo 1.1.514, dcp(name) does not create it: open 'dcp' and type the name, "
                  "or use dcp.new from Python")
            return 1
        print('Project exist in disk: "%s". Opening it.' % name)
        return _run(name)
    return _run(None)


if __name__ == "__main__":
    sys.exit(main())

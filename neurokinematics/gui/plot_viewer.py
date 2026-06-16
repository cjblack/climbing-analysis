"""
Plot viewer panel for the neurokinematics GUI.

Embeds a matplotlib figure inside a dock widget.
Provides controls to select what to plot from loaded sessions/groups.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSizePolicy, QGroupBox, QFormLayout,
    QLineEdit, QFileDialog, QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QAbstractItemView
)
from PySide6.QtCore import Qt

# matplotlib backend for embedding in Qt
import warnings
import matplotlib
try:
    matplotlib.use('QtAgg')
except ImportError:
    # Headless environments (e.g. CI) have no Qt event loop, so switching to the
    # interactive QtAgg backend fails. Leave the default backend in place so the
    # module can still be imported for its non-GUI helpers.
    pass
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from neurokinematics.gui.style import brand_cmap, BG_MID

# Constrained layout occasionally can't fit a very dense ArviZ grid into a
# narrow dock and emits this warning; the plot still renders, so keep it quiet.
warnings.filterwarnings("ignore", message=r".*constrained_layout not applied.*")


def find_latest_glm_predictions(models_dir, glm_type=None):
    """Return the most-recently-modified ``predictions.zarr`` under a session's
    ``models/glm`` tree, or None.

    GLM fits are saved by ``save_glm_results`` under
    ``<models>/glm/<type>/<run>/predictions.zarr`` (model-comparison runs nest one
    more level under the model name), so this globs recursively and picks the
    newest. When ``glm_type`` is given (e.g. ``'encoder'`` / ``'decoder'``) the
    search is restricted to that subtree. Pure/Qt-free so it can be unit-tested.
    """
    if not models_dir:
        return None
    base = Path(models_dir) / 'glm'
    if glm_type:
        base = base / glm_type
    if not base.exists():
        return None
    candidates = list(base.glob('**/predictions.zarr'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class PlotCanvas(QWidget):
    """Embeds a matplotlib Figure with navigation toolbar.

    Figures use the 'constrained' layout engine, which re-flows panels so
    multi-panel plots (e.g. the ArviZ diagnostics) stay cleanly separated. The
    canvas sits in a scroll area: most plots fill the panel, but a *tall* plot
    (e.g. a trace plot with many variables) keeps enough height for its rows and
    gets a vertical-only scrollbar instead of collapsing into overlap. The plot
    area itself is white; only the empty canvas is dark-filled so it doesn't
    clash with the dark UI before anything is plotted.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.fig    = Figure(figsize=(6, 4), layout='constrained')
        self.fig.set_facecolor(BG_MID)          # dark until something is plotted
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background: transparent; border: none;")

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        # never scroll horizontally — figures always fit the panel width
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self.canvas)

        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self._scroll)

    def clear(self):
        self.fig.clear()
        self.fig.set_facecolor(BG_MID)          # back to the dark empty state
        self.canvas.draw_idle()

    def get_ax(self):
        """Reset to a single fresh white axes (for the simple, non-ArviZ plots)."""
        try:
            self.fig.set_layout_engine('constrained')
        except Exception:
            pass
        self.canvas.setMinimumHeight(0)         # fill the panel, no scroll
        self.fig.clear()
        self.fig.set_facecolor('white')
        self.fig.set_size_inches(6, 4)
        return self.fig.add_subplot(111)

    def set_figure(self, fig, vscroll: bool = False):
        """Host an externally created figure (e.g. from ArviZ) in this widget.

        The figure is switched to the 'constrained' layout engine and keeps its
        white background. If *vscroll* is True the canvas keeps the figure's
        natural height (so tall plots stay readable and scroll vertically rather
        than overlapping); otherwise it fills the panel.
        """
        try:
            fig.set_layout_engine('constrained')
        except Exception:
            pass

        # tear down the existing canvas + toolbar
        old_canvas = self._scroll.takeWidget()
        self._layout.removeWidget(self.toolbar)
        self.toolbar.deleteLater()
        if old_canvas is not None:
            old_canvas.deleteLater()

        self.fig    = fig
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if vscroll:
            # pin the natural height so rows keep their room; width still fills
            dpi  = fig.get_dpi()
            h_px = int(fig.get_size_inches()[1] * dpi)
            self.canvas.setMinimumHeight(max(1, h_px))
        else:
            self.canvas.setMinimumHeight(0)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background: transparent; border: none;")

        self._layout.insertWidget(0, self.toolbar)
        self._scroll.setWidget(self.canvas)
        self.canvas.draw_idle()


class PlotViewerPanel(QWidget):
    """
    Interactive plot viewer.
    Lets the user pick a loaded group/subject and a plot type,
    then renders the result in an embedded matplotlib figure.
    """

    PLOT_TYPES = [
        "── Group / Subject ──",
        "Velocity distribution",
        "Session velocity trend",
        "Movement event rasters",
        "LFP power spectrum",
        "── Session ──",
        "Trajectories",
        "Velocity traces",
        "── Spikes ──",
        "Waveforms",
        "Autocorrelograms",
        "Spike rasters",
        "── Encoding / Decoding ──",
        "GLM encoder fit",
        "GLM decoder fit",
        "── Analysis results ──",
        "Trace plot",
        "Forest plot",
        "Posterior plot",
        "Posterior predictive",
    ]

    # plots that need a session source
    SESSION_PLOTS = {"Trajectories", "Velocity traces"}
    # plots that need a node selector
    NODE_PLOTS    = {"Trajectories", "Velocity traces", "Velocity distribution"}
    # spike plots (need a unit selector); rasters also need node + movement event
    SPIKE_PLOTS   = {"Waveforms", "Autocorrelograms", "Spike rasters"}
    RASTER_PLOTS  = {"Spike rasters"}
    # plots that need a trace file
    ANALYSIS_PLOTS = {"Trace plot", "Forest plot", "Posterior plot", "Posterior predictive"}
    # GLM encoder/decoder result plots (need a session source with a models dir)
    ENCODING_PLOTS = {"GLM encoder fit", "GLM decoder fit"}

    def __init__(self, loaded_objects: dict, log, parent=None):
        super().__init__(parent)
        self._objects = loaded_objects
        self._log     = log
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Controls ──
        ctrl = QGroupBox("Plot Settings")
        form = QFormLayout(ctrl)
        form.setSpacing(8)

        self._source_combo = QComboBox()
        self._refresh_sources()
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Source:", self._source_combo)

        self._plot_combo = QComboBox()
        for pt in self.PLOT_TYPES:
            self._plot_combo.addItem(pt)
            if pt.startswith("──"):
                idx = self._plot_combo.count() - 1
                from PySide6.QtCore import Qt as _Qt
                self._plot_combo.model().item(idx).setEnabled(False)
        self._plot_combo.currentTextChanged.connect(self._on_plot_type_changed)
        form.addRow("Plot type:", self._plot_combo)

        # node selector (shown for node-specific plots)
        self._node_row = QWidget()
        node_layout = QHBoxLayout(self._node_row)
        node_layout.setContentsMargins(0, 0, 0, 0)
        self._node_combo = QComboBox()
        self._node_combo.setMinimumWidth(140)
        node_layout.addWidget(self._node_combo)
        node_layout.addStretch()
        form.addRow("Node:", self._node_row)
        self._node_row.setVisible(False)

        # movement-event selector (spike rasters)
        self._event_row = QWidget()
        event_layout = QHBoxLayout(self._event_row)
        event_layout.setContentsMargins(0, 0, 0, 0)
        self._event_combo = QComboBox()
        self._event_combo.setMinimumWidth(140)
        event_layout.addWidget(self._event_combo)
        event_layout.addStretch()
        form.addRow("Movement event:", self._event_row)
        self._event_row.setVisible(False)

        # unit selector (spike plots) — multi-select
        self._units_row = QWidget()
        units_layout = QVBoxLayout(self._units_row)
        units_layout.setContentsMargins(0, 0, 0, 0)
        self._units_list = QListWidget()
        self._units_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._units_list.setMaximumHeight(90)
        units_hint = QLabel("Select one or more units (none = first few)")
        units_hint.setObjectName("subheading")
        units_layout.addWidget(self._units_list)
        units_layout.addWidget(units_hint)
        form.addRow("Units:", self._units_row)
        self._units_row.setVisible(False)

        # trace file picker
        self._trace_row = QWidget()
        trace_layout = QHBoxLayout(self._trace_row)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        self._trace_edit = QLineEdit()
        self._trace_edit.setPlaceholderText("Path to saved trace (.pkl) — or leave blank to use last run...")
        trace_browse = QPushButton("Browse")
        trace_browse.setObjectName("secondary")
        trace_browse.setFixedWidth(70)
        trace_browse.clicked.connect(self._browse_trace)
        trace_layout.addWidget(self._trace_edit)
        trace_layout.addWidget(trace_browse)
        form.addRow("Trace file:", self._trace_row)
        self._trace_row.setVisible(False)

        layout.addWidget(ctrl)

        # ── Canvas (created before buttons so clear_btn can reference it) ──
        self.canvas_widget = PlotCanvas()

        btn_row = QHBoxLayout()
        plot_btn = QPushButton("▶  Plot")
        plot_btn.setObjectName("run")
        plot_btn.clicked.connect(self._plot)
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self.canvas_widget.clear)
        btn_row.addWidget(plot_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(self.canvas_widget, stretch=1)

    def _refresh_sources(self):
        self._source_combo.clear()
        for key, obj in self._objects.items():
            from neurokinematics.data.group import ExperimentGroup
            from neurokinematics.data.subject import ExperimentSubject
            if isinstance(obj, ExperimentGroup):
                name = obj.group_id
                self._source_combo.addItem(f"👥 {name}", userData=obj)
                # also add each subject's sessions
                for subj in obj.subjects:
                    for sess in (subj.sessions or []):
                        sess_id = getattr(sess, 'session_id', str(sess))
                        self._source_combo.addItem(
                            f"  📅 {subj.subject_id} / {sess_id}",
                            userData=sess
                        )
            elif isinstance(obj, ExperimentSubject):
                name = obj.subject_id
                self._source_combo.addItem(f"🐭 {name}", userData=obj)
                for sess in (obj.sessions or []):
                    sess_id = getattr(sess, 'session_id', str(sess))
                    self._source_combo.addItem(
                        f"  📅 {name} / {sess_id}",
                        userData=sess
                    )

    def _on_source_changed(self, _idx: int):
        """Repopulate selectors for the chosen source and current plot type."""
        plot_type = self._plot_combo.currentText()
        if plot_type in self.SPIKE_PLOTS:
            self._refresh_spike_controls()
        else:
            self._refresh_pose_nodes()

    def _refresh_pose_nodes(self):
        """Populate the node combo from the selected session's pose data."""
        obj = self._source_combo.currentData()
        self._node_combo.clear()
        dirs = getattr(obj, 'dirs', {})
        pose_dir = dirs.get('pose', None)
        if pose_dir:
            pose_file = Path(pose_dir) / 'pose_data.csv'
            if pose_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(pose_file, nrows=0)
                    nodes = sorted(set(
                        c.rsplit('_', 1)[0] for c in df.columns
                        if c.lower().endswith(('_x', '_y'))
                    ))
                    if nodes:
                        self._node_combo.addItems(nodes)
                        return
                except Exception:
                    pass
        # fallback: common node names
        self._node_combo.addItems(['r_forepaw', 'l_forepaw', 'r_hindpaw', 'l_hindpaw'])

    # ── spike controls ────────────────────────────────────────────────────────
    def _spikes_dir(self, obj):
        return getattr(obj, 'dirs', {}).get('spikes', None)

    def _raster_pkl(self, obj):
        sd = self._spikes_dir(obj)
        if not sd:
            return None
        p = Path(sd) / 'rasters' / 'movement_aligned_rasters.pkl'
        return p if p.exists() else None

    def _load_analyzer(self, obj):
        """Load (and cache) the SpikeInterface sorting analyzer for a session."""
        sd = self._spikes_dir(obj)
        folder = Path(sd) / 'sorting_analyzer' if sd else None
        if not folder or not folder.exists():
            raise FileNotFoundError(
                "No sorting_analyzer for this session — run spike sorting first."
            )
        if not hasattr(self, '_analyzer_cache'):
            self._analyzer_cache = {}
        key = str(folder)
        if key not in self._analyzer_cache:
            import spikeinterface as si
            self._analyzer_cache[key] = si.load_sorting_analyzer(folder)
        return self._analyzer_cache[key]

    def _available_units(self, obj, plot_type):
        try:
            if plot_type in self.RASTER_PLOTS:
                pkl = self._raster_pkl(obj)
                if pkl is None:
                    return []
                import pandas as pd
                return list(pd.read_pickle(pkl)['unit_id'].unique())
            analyzer = self._load_analyzer(obj)
            return list(analyzer.unit_ids)
        except Exception as e:
            self._log.log(f"Could not read spike units: {e}", 'warning')
            return []

    def _refresh_spike_controls(self):
        from PySide6.QtCore import Qt as _Qt
        obj       = self._source_combo.currentData()
        plot_type = self._plot_combo.currentText()

        # units
        self._units_list.clear()
        for u in self._available_units(obj, plot_type):
            item = QListWidgetItem(str(u))
            item.setData(_Qt.UserRole, u)
            self._units_list.addItem(item)

        # raster-only: node + movement event from the raster table
        if plot_type in self.RASTER_PLOTS:
            self._node_combo.clear()
            self._event_combo.clear()
            pkl = self._raster_pkl(obj)
            if pkl is not None:
                try:
                    import pandas as pd
                    df = pd.read_pickle(pkl)
                    self._node_combo.addItems([str(n) for n in df['node'].unique()])
                    self._event_combo.addItems([str(e) for e in df['movement_event'].unique()])
                except Exception:
                    pass

    def _selected_units(self):
        from PySide6.QtCore import Qt as _Qt
        return [i.data(_Qt.UserRole) for i in self._units_list.selectedItems()]

    @staticmethod
    def _widget_figure(widget):
        import matplotlib.pyplot as plt
        fig = getattr(widget, 'figure', None)
        if fig is None:
            ax = getattr(widget, 'ax', None)
            fig = ax.get_figure() if ax is not None else plt.gcf()
        return fig

    def _plot(self):
        obj       = self._source_combo.currentData()
        plot_type = self._plot_combo.currentText()

        if plot_type.startswith("──"):
            return

        if plot_type in self.ANALYSIS_PLOTS:
            try:
                self._plot_analysis_result(plot_type)
            except Exception as e:
                self._log.log(f"Plot error: {e}", 'error')
            return

        if obj is None:
            self._log.log("No source selected.", 'warning')
            return

        if plot_type in self.ENCODING_PLOTS:
            try:
                ax = self.canvas_widget.get_ax()
                glm_type = 'decoder' if 'decoder' in plot_type.lower() else 'encoder'
                self._plot_glm_fit(obj, ax, glm_type=glm_type)
                self.canvas_widget.canvas.draw()
            except Exception:
                import traceback
                self._log.log(f"Plot error: {traceback.format_exc()}", 'error')
            return

        if plot_type in self.SPIKE_PLOTS:
            try:
                self._plot_spikes(obj, plot_type)
            except Exception:
                import traceback
                self._log.log(f"Plot error: {traceback.format_exc()}", 'error')
            return

        node = self._node_combo.currentText() if self._node_row.isVisible() else None

        try:
            ax = self.canvas_widget.get_ax()
            self._dispatch(obj, plot_type, ax, node=node)
            self.canvas_widget.canvas.draw()
        except Exception as e:
            import traceback
            self._log.log(f"Plot error: {traceback.format_exc()}", 'error')

    def _plot_spikes(self, obj, plot_type):
        units = self._selected_units()
        if plot_type == "Waveforms":
            self._plot_waveforms(obj, units)
        elif plot_type == "Autocorrelograms":
            self._plot_autocorrelograms(obj, units)
        elif plot_type == "Spike rasters":
            ax = self.canvas_widget.get_ax()
            node  = self._node_combo.currentText() or None
            event = self._event_combo.currentText() or None
            self._plot_spike_rasters(obj, ax, units, node, event)
            self.canvas_widget.canvas.draw()

    def _plot_waveforms(self, obj, units):
        import spikeinterface.widgets as sw
        analyzer = self._load_analyzer(obj)
        uids = units or list(analyzer.unit_ids)[:6]
        w = sw.plot_unit_waveforms(analyzer, unit_ids=uids, backend='matplotlib')
        fig = self._widget_figure(w)
        self.canvas_widget.set_figure(fig, vscroll=len(uids) > 4)
        self._log.log(f"Waveforms — units {list(uids)}", 'success')

    def _plot_autocorrelograms(self, obj, units):
        import spikeinterface.widgets as sw
        analyzer = self._load_analyzer(obj)
        uids = units or list(analyzer.unit_ids)[:6]
        w = sw.plot_autocorrelograms(analyzer.sorting, unit_ids=uids, backend='matplotlib')
        fig = self._widget_figure(w)
        self.canvas_widget.set_figure(fig, vscroll=len(uids) > 4)
        self._log.log(f"Autocorrelograms — units {list(uids)}", 'success')

    def _plot_spike_rasters(self, obj, ax, units, node, event):
        """Movement-aligned spike rasters from movement_aligned_rasters.pkl."""
        import pandas as pd
        import numpy as np
        pkl = self._raster_pkl(obj)
        if pkl is None:
            raise ValueError("No movement_aligned_rasters.pkl — align movements first.")
        df = pd.read_pickle(pkl)
        if node:
            df = df[df['node'].astype(str) == str(node)]
        if event:
            df = df[df['movement_event'].astype(str) == str(event)]
        if not units:
            units = list(df['unit_id'].unique())[:5]

        cmap = brand_cmap()
        positions, colours, yticks, ylabels = [], [], [], []
        offset = 0
        for i, uid in enumerate(units):
            usub = df[df['unit_id'] == uid]
            colour = cmap(0.5 if len(units) == 1 else i / (len(units) - 1))
            start = offset
            for arr in usub['spike_raster']:
                positions.append(np.asarray(arr, dtype=float))
                colours.append(colour)
                offset += 1
            if offset > start:
                yticks.append((start + offset) / 2.0)
                ylabels.append(f"unit {uid}")

        if positions:
            ax.eventplot(positions, colors=colours,
                         lineoffsets=list(range(len(positions))), linelengths=0.8)
        ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.6)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Time from movement event (s)")
        ax.set_ylabel("Unit / trial")
        ax.set_title(f"Spike rasters — {node or 'all nodes'} / {event or 'all events'}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_glm_fit(self, obj, ax, glm_type='encoder'):
        """Observed vs (cross-validated) GLM-predicted signal over the event window,
        averaged across events, from the most recent ``predictions.zarr`` of the
        requested type ('encoder' = unit firing; 'decoder' = a movement feature)."""
        import numpy as np
        from neurokinematics.io import load_zarr

        models_dir = getattr(obj, 'dirs', {}).get('models')
        pred_path = find_latest_glm_predictions(models_dir, glm_type=glm_type)
        if pred_path is None:
            # self-diagnose: report what (if anything) is actually under models/glm
            glm_root = Path(models_dir) / 'glm' if models_dir else None
            if not models_dir:
                where = "this source has no 'models' directory — is the Plot Viewer source the session you fit on (a 📅 row, not the group/subject)?"
            elif not (glm_root and glm_root.exists()):
                where = f"nothing under {glm_root} — no GLM has been fit/saved for this session yet."
            else:
                types = sorted(p.name for p in glm_root.iterdir() if p.is_dir())
                where = (f"found GLM types {types} but none named '{glm_type}'. "
                         "If the fit is still running, wait for 'Done.' in the log; "
                         "if it errored, the fit logged a '✗' line — check the log panel.")
            raise ValueError(f"No GLM {glm_type} predictions found — {where}")

        ds = load_zarr(pred_path, method='xarray')
        obs = np.asarray(ds['observed_counts'].values, dtype=float)
        pred = np.asarray(ds['predicted_counts'].values, dtype=float)
        valid = ds['valid'].values if 'valid' in ds else np.isfinite(obs)
        t = ds['time_bin'].values

        obs_v = np.where(valid, obs, np.nan)
        pred_v = np.where(valid, pred, np.nan)
        obs_m = np.nanmean(obs_v, axis=0)
        pred_m = np.nanmean(pred_v, axis=0)
        n = np.clip(np.sum(valid & np.isfinite(pred_v), axis=0), 1, None)
        obs_se = np.nanstd(obs_v, axis=0) / np.sqrt(n)

        # best-effort: shade the pre-movement portion of the window
        self._shade_pre_movement(obj, ax, t)

        attrs = dict(ds.attrs or {})
        is_decoder = str(attrs.get('model_type', glm_type)) == 'decoder'

        params_yaml = self._load_glm_params(pred_path)
        metrics = params_yaml.get('metrics', {}) or {}
        family = params_yaml.get('family') or ('Gaussian' if is_decoder else 'Poisson')
        cross_validated = bool(metrics.get('cross_validated', False))

        # goodness-of-fit computed from the predictions actually plotted — these are
        # held-out predictions when CV ran, in-sample otherwise — so the score on
        # screen always matches the curve being shown
        from neurokinematics.models.glm import glm_cv_scores
        fit_mask = valid & np.isfinite(obs) & np.isfinite(pred)
        score = glm_cv_scores(obs[fit_mask], pred[fit_mask], family) if fit_mask.any() else {}
        r2, corr = score.get('cv_r2'), score.get('cv_corr')
        r2_name = 'R²' if str(family).lower() == 'gaussian' else 'pseudo-R²'
        kind = 'CV' if cross_validated else 'in-sample'
        score_str = (f"{kind} {r2_name} = {r2:.3f}"
                     if (r2 is not None and np.isfinite(r2)) else f"{r2_name} = n/a")

        pred_label = 'GLM predicted' + (' (CV)' if cross_validated else '')
        ax.plot(t, obs_m, color='black', lw=1.5, label='Observed')
        ax.fill_between(t, obs_m - obs_se, obs_m + obs_se, color='black', alpha=0.15)
        ax.plot(t, pred_m, color='#9e1e62', lw=2, ls='--', label=pred_label)

        node = attrs.get('node', '?')
        ax.set_xlabel("Time in event window (s)")
        if is_decoder:
            target = attrs.get('target') or (attrs.get('features', {}) or {}).get('pose', 'feature')
            units = attrs.get('unit', [])
            n_units = len(units) if isinstance(units, (list, tuple)) else 1
            ax.set_title(f"GLM decoder — {n_units} unit(s) → {node} {target}\n{score_str}")
            ax.set_ylabel(str(target))
        else:
            unit = attrs.get('unit', '?')
            feats = attrs.get('features', {})
            pose_feats = feats.get('pose') if isinstance(feats, dict) else feats
            basis = attrs.get('basis')
            ax.set_title(f"GLM encoder — {node} → unit {unit}\n{score_str}")
            ax.set_ylabel("Spike count / bin")
            extra = f"features: {pose_feats}"
            if isinstance(basis, dict):
                extra += f"   ·   basis {basis.get('window')} ({basis.get('n_basis')} fns)"
            ax.text(0.01, 0.99, extra, transform=ax.transAxes, va='top', ha='left',
                    fontsize=7, color='gray')

        # fit-detail box (Pearson r + validation scheme), bottom-right
        detail = []
        if corr is not None and np.isfinite(corr):
            detail.append(f"r = {corr:.3f}")
        if cross_validated:
            ns = metrics.get('n_splits')
            detail.append(f"{ns}-fold CV (grouped by event)" if ns else "event-grouped CV")
        else:
            detail.append("in-sample (no CV)")
        if metrics.get('shuffle_p') is not None:
            detail.append(f"p = {metrics['shuffle_p']:.3g} vs shuffle")
        if detail:
            ax.text(0.99, 0.02, "\n".join(detail), transform=ax.transAxes, va='bottom',
                    ha='right', fontsize=7.5, color='#262161',
                    bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.85))

        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._log.log(f"GLM {'decoder' if is_decoder else 'encoder'} fit — "
                      f"{pred_path.parent.name}", 'success')

    @staticmethod
    def _load_glm_params(pred_path):
        """Read the sibling glm_params.yaml (family, metrics, cv scheme) if present."""
        try:
            import yaml
            pj = Path(pred_path).parent / 'glm_params.yaml'
            if pj.exists():
                with open(pj) as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    def _shade_pre_movement(self, obj, ax, t):
        """Lightly shade the pre-movement portion of the window if the binned spike
        store carries the ``pre_movement`` mask. Best-effort; silent on failure."""
        try:
            import numpy as np
            from neurokinematics.io import load_zarr
            spikes_dir = getattr(obj, 'dirs', {}).get('spikes')
            if not spikes_dir:
                return
            stores = sorted(Path(spikes_dir).glob('movement_spike_counts_*ms.zarr'))
            if not stores or 'pre_movement' not in (sds := load_zarr(stores[-1], method='xarray')):
                return
            frac = sds['pre_movement'].values.mean(axis=0)   # per-bin fraction pre-movement
            tb = sds['time_bin'].values
            pre_bins = tb[frac >= 0.5]
            if pre_bins.size:
                ax.axvspan(float(tb.min()), float(pre_bins.max()),
                           color='steelblue', alpha=0.08, label='pre-movement')
        except Exception:
            pass

    def _on_plot_type_changed(self, plot_type: str):
        is_spike = plot_type in self.SPIKE_PLOTS
        is_raster = plot_type in self.RASTER_PLOTS
        self._trace_row.setVisible(plot_type in self.ANALYSIS_PLOTS)
        self._node_row.setVisible(plot_type in self.NODE_PLOTS or is_raster)
        self._event_row.setVisible(is_raster)
        self._units_row.setVisible(is_spike)
        if is_spike:
            self._refresh_spike_controls()
        elif plot_type in self.NODE_PLOTS:
            self._refresh_pose_nodes()

    def _browse_trace(self):
        # default to the results/bayesian folder of the first loaded group
        start = ""
        for obj in self._objects.values():
            results_dir = getattr(obj, 'dirs', {}).get('results', None)
            if results_dir:
                bayesian_dir = Path(results_dir) / 'bayesian'
                if bayesian_dir.exists():
                    start = str(bayesian_dir)
                break
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Saved Trace", start,
            "Pickle files (*.pkl);;All files (*)"
        )
        if path:
            self._trace_edit.setText(path)

    def _load_trace(self):
        import pickle
        path = self._trace_edit.text().strip()

        if not path:
            # try to use in-memory trace from last analysis run
            obj = self._source_combo.currentData()
            if obj is not None and hasattr(obj, '_last_trace'):
                return obj._last_trace
            raise ValueError(
                "No trace available. Run an analysis first, or browse to a saved .pkl file."
            )

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        with open(p, 'rb') as f:
            data = pickle.load(f)

        # handle both raw trace and our wrapped dict format
        if isinstance(data, dict) and 'trace' in data:
            return data['trace']
        return data

    def _dispatch(self, obj, plot_type: str, ax, node: str = None):
        if plot_type == "Velocity distribution":
            self._plot_velocity_dist(obj, ax, node=node)
        elif plot_type == "Session velocity trend":
            self._plot_velocity_trend(obj, ax)
        elif plot_type == "Movement event rasters":
            self._plot_rasters(obj, ax)
        elif plot_type == "LFP power spectrum":
            self._plot_lfp_spectrum(obj, ax)
        elif plot_type == "Trajectories":
            self._plot_trajectories(obj, ax, node=node)
        elif plot_type == "Velocity traces":
            self._plot_velocity_traces(obj, ax, node=node)

    def _plot_velocity_dist(self, obj, ax, node: str = None):
        """Velocity distribution from group pose summary."""
        import pandas as pd

        df = self._load_pose_summary(obj)
        if df is None:
            raise ValueError("No pose summary found. Run group.summarize('pose') first.")

        col = 'v_mag_max' if 'v_mag_max' in df.columns else df.select_dtypes('number').columns[0]

        if node and 'node' in df.columns:
            df = df[df['node'] == node]
            ax.hist(df[col].dropna(), bins=40, alpha=0.7, color='teal', density=True)
            ax.set_title(f"Velocity Distribution — {node}")
        else:
            nodes = df['node'].unique() if 'node' in df.columns else []
            for n in nodes[:4]:
                subset = df[df['node'] == n][col].dropna()
                ax.hist(subset, bins=40, alpha=0.5, label=n, density=True)
            ax.legend(fontsize=8)
            ax.set_title("Velocity Distribution")

        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_velocity_trend(self, obj, ax):
        """Mean velocity per session with per-subject lines."""
        import numpy as np

        df = self._load_pose_summary(obj)
        if df is None:
            raise ValueError("No pose summary found. Run group.summarize('pose') first.")

        col = 'v_mag_max' if 'v_mag_max' in df.columns else df.select_dtypes('number').columns[0]
        node = 'r_forepaw' if 'r_forepaw' in df.get('node', {}).values else \
               (df['node'].iloc[0] if 'node' in df.columns else None)

        sub = df[df['node'] == node] if node else df

        for subj_id in sub.get('id', sub.get('subject_id', [])).unique() if 'id' in sub.columns or 'subject_id' in sub.columns else []:
            id_col = 'id' if 'id' in sub.columns else 'subject_id'
            s = sub[sub[id_col] == subj_id].groupby('session_number')[col].mean()
            ax.plot(s.index, s.values, color='teal', alpha=0.3, lw=1)

        group_mean = sub.groupby('session_number')[col].mean()
        ax.plot(group_mean.index, group_mean.values, color='#9e1e62', lw=2.5, label='Group mean')

        ax.set_xlabel("Session")
        ax.set_ylabel(f"Mean {col}")
        ax.set_title(f"Velocity Trend — {node or 'all nodes'}")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_rasters(self, obj, ax):
        """Movement-aligned spike rasters for first available session."""
        from neurokinematics.data.group import ExperimentGroup
        from neurokinematics.data.subject import ExperimentSubject

        sessions = []
        if isinstance(obj, ExperimentGroup):
            for subj in obj.subjects:
                if subj.sessions:
                    sessions.extend(subj.sessions)
        elif isinstance(obj, ExperimentSubject):
            sessions = obj.sessions or []

        for sess in sessions:
            raster_dir = getattr(sess, 'dirs', {}).get('spikes', None)
            if not raster_dir:
                continue
            raster_files = list(Path(raster_dir).glob("*raster*.csv"))
            if raster_files:
                import pandas as pd
                df = pd.read_csv(raster_files[0])
                self._draw_raster(df, ax)
                ax.set_title(f"Spike Raster — {getattr(sess, 'session_id', '')}")
                return

        raise ValueError("No raster files found in any session.")

    def _draw_raster(self, df, ax):
        """Generic raster plot from a spike times dataframe."""
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial / Unit")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if df.empty:
            ax.text(0.5, 0.5, "Empty raster", ha='center', transform=ax.transAxes)
            return
        # assume columns: time, trial (or unit)
        t_col = df.columns[0]
        y_col = df.columns[1] if len(df.columns) > 1 else None
        if y_col:
            ax.scatter(df[t_col], df[y_col], s=1, color='#9e1e62', alpha=0.6)
        else:
            ax.eventplot(df[t_col].values, colors='#9e1e62', linewidths=0.5)

    def _plot_lfp_spectrum(self, obj, ax):
        """Power spectrum from first available LFP output."""
        from neurokinematics.data.group import ExperimentGroup
        from neurokinematics.data.subject import ExperimentSubject
        import numpy as np
        from scipy import signal as sig

        sessions = []
        if isinstance(obj, ExperimentGroup):
            for subj in obj.subjects:
                if subj.sessions:
                    sessions.extend(subj.sessions)
        elif isinstance(obj, ExperimentSubject):
            sessions = obj.sessions or []

        for sess in sessions:
            lfp_dir = getattr(sess, 'dirs', {}).get('lfp', None)
            if not lfp_dir or not Path(lfp_dir).exists():
                continue
            zarr_dirs = list(Path(lfp_dir).glob("*.zarr"))
            if zarr_dirs:
                from neurokinematics.io import load_zarr
                data, attrs = load_zarr(zarr_dirs[0])
                fs = attrs.get('fs', 1000)
                # take first channel, first chunk
                chunk = data[0, :min(fs * 10, data.shape[1])]
                f, pxx = sig.welch(chunk, fs=fs, nperseg=min(1024, len(chunk)))
                ax.semilogy(f, pxx, color='#262161', lw=1.2)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD")
                ax.set_title(f"LFP Power Spectrum — {getattr(sess, 'session_id', '')}")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                return

        raise ValueError("No LFP zarr files found.")

    # smaller fonts so the dense ArviZ panels don't bunch up in the narrow dock
    _ARVIZ_RC = {
        'font.size':       7,
        'axes.titlesize':  8,
        'axes.labelsize':  7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
    }

    def _plot_analysis_result(self, plot_type: str):
        """Render an ArviZ diagnostic plot and host its figure in the canvas."""
        import arviz as az
        import matplotlib as mpl

        trace = self._load_trace()

        # build the plot under reduced font sizes; the text artists keep these
        # sizes so constrained layout can pack the panels without overlap
        with mpl.rc_context(self._ARVIZ_RC):
            if plot_type == "Trace plot":
                axes = az.plot_trace(trace, backend='matplotlib')
            elif plot_type == "Forest plot":
                # combined=False keeps each chain as its own row so all chains
                # are visible (combined=True pools them into a single interval)
                axes = az.plot_forest(trace, backend='matplotlib', combined=False)
            elif plot_type == "Posterior plot":
                axes = az.plot_posterior(trace, backend='matplotlib')
            elif plot_type == "Posterior predictive":
                if not hasattr(trace, 'posterior_predictive'):
                    raise ValueError("Trace has no posterior_predictive group.\n"
                                     "Run pm.sample_posterior_predictive() first.")
                axes = az.plot_ppc(trace, backend='matplotlib')
            else:
                return

            # ArviZ returns an ndarray / list of axes; grab their parent figure
            fig = self._figure_from_axes(axes)

        # trace / forest grids grow tall with many variables — keep their height
        # and scroll vertically rather than squashing the rows together
        vscroll = plot_type in ("Trace plot", "Forest plot")
        self.canvas_widget.set_figure(fig, vscroll=vscroll)
        self._log.log(f"{plot_type} rendered.", 'success')

    @staticmethod
    def _figure_from_axes(axes):
        """Return the parent Figure from whatever an ArviZ plot returned."""
        import numpy as np
        import matplotlib.pyplot as plt

        if hasattr(axes, 'flat'):
            candidates = list(axes.flat)
        elif isinstance(axes, (list, tuple)):
            candidates = list(np.ravel(np.asarray(axes, dtype=object)))
        else:
            candidates = [axes]

        for a in candidates:
            if hasattr(a, 'get_figure'):
                fig = a.get_figure()
                if fig is not None:
                    return fig
        return plt.gcf()

    def _load_movement_features(self, session):
        """
        Load movement_features.zarr from a session as an xarray Dataset.
        Returns (ds, node_list) or raises.
        """
        dirs     = getattr(session, 'dirs', {})
        pose_dir = dirs.get('pose', None)
        if not pose_dir:
            raise ValueError("Session has no pose directory.")
        zarr_path = Path(pose_dir) / 'movement_features.zarr'
        if not zarr_path.exists():
            raise FileNotFoundError(
                f"movement_features.zarr not found in {pose_dir}.\n"
                "Run session.process('pose') first."
            )
        from neurokinematics.io import load_zarr
        ds = load_zarr(zarr_path, method='xarray')
        nodes = list(ds.coords['node'].values)
        return ds, nodes

    def _plot_trajectories(self, session, ax, node: str = 'r_forepaw'):
        """
        Plot each movement trajectory (x,y position relative to start = 0,0)
        for a given node, loaded from movement_features.zarr.

        Dimensions: ds['position'] = [event, time, node, coord]
        coord = ['x', 'y']
        """
        import numpy as np

        ds, nodes = self._load_movement_features(session)
        node      = node if node in nodes else nodes[0]
        sess_id   = getattr(session, 'session_id', '')

        # shape: [event, time, coord]
        pos = ds['position'].sel(node=node).values   # (n_events, n_time, 2)
        valid = ds['valid'].values                   # (n_events, n_time)

        n_events = pos.shape[0]
        colours  = brand_cmap()(np.linspace(0.1, 0.9, n_events))

        for i in range(n_events):
            mask = valid[i]
            x = pos[i, mask, 0]
            y = pos[i, mask, 1]
            if len(x) < 2:
                continue
            # centre at start position
            x = x - x[0]
            y = y - y[0]
            ax.plot(x, y, color=colours[i], alpha=0.5, lw=0.8)
            # mark start
            ax.scatter(0, 0, color=colours[i], s=10, zorder=3, alpha=0.7)

        ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
        ax.set_xlabel("Δx (px)")
        ax.set_ylabel("Δy (px)")
        ax.set_title(f"Movement Trajectories — {node}  [{sess_id}]  "
                     f"(n={n_events})")
        ax.set_aspect('equal', adjustable='datalim')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_velocity_traces(self, session, ax, node: str = 'r_forepaw'):
        """
        Plot speed traces for all movements, each aligned to the index of
        peak speed (t=0), loaded from movement_features.zarr.

        Dimensions: ds['speed'] = [event, time, node]
        """
        import numpy as np

        ds, nodes = self._load_movement_features(session)
        node      = node if node in nodes else nodes[0]
        sess_id   = getattr(session, 'session_id', '')

        # shape: [event, time]
        speed = ds['speed'].sel(node=node).values   # (n_events, n_time)
        valid = ds['valid'].values                   # (n_events, n_time)

        n_events = speed.shape[0]
        colours  = brand_cmap()(np.linspace(0.1, 0.9, n_events))

        all_aligned = []
        half_win    = 75   # frames either side of peak

        for i, col in zip(range(n_events), colours):
            v    = speed[i]
            mask = valid[i]
            v    = np.where(mask, v, np.nan)
            if np.all(np.isnan(v)):
                continue

            peak_idx = int(np.nanargmax(v))
            start    = max(0, peak_idx - half_win)
            end      = min(len(v), peak_idx + half_win)
            t        = np.arange(start - peak_idx, end - peak_idx)

            ax.plot(t, v[start:end], color=col, alpha=0.3, lw=0.7)

            # build aligned array for mean
            buf = np.full(2 * half_win, np.nan)
            i_start = half_win - (peak_idx - start)
            i_end   = i_start + (end - start)
            buf[i_start:i_end] = v[start:end]
            all_aligned.append(buf)

        if all_aligned:
            mean_v = np.nanmean(all_aligned, axis=0)
            t_full = np.arange(-half_win, half_win)
            ax.plot(t_full, mean_v, color='#9e1e62', lw=2.5,
                    label='Mean', zorder=5)

        ax.axvline(0, color='gray', linestyle='--', lw=1, alpha=0.5,
                   label='Peak')
        ax.set_xlabel("Frames from peak speed")
        ax.set_ylabel("Speed (px/frame)")
        ax.set_title(f"Velocity Traces — {node}  [{sess_id}]  "
                     f"(n={n_events})")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _load_pose_summary(self, obj):
        """Load pose_metrics.parquet from a group or subject's group."""
        from neurokinematics.data.group import ExperimentGroup
        import pandas as pd

        if isinstance(obj, ExperimentGroup):
            summaries_dir = obj.dirs.get('summaries', None)
            if summaries_dir:
                p = Path(summaries_dir) / 'pose_metrics.parquet'
                if p.exists():
                    return pd.read_parquet(p)
        return None

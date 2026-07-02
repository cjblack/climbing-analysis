"""Session management and workflow orchestration for neurokinematics.

This module provides the core class, which serves as a high-level interface for organising, preprocessing, and aligning electrophysiology and markerless pose data within a single experimental session.

The session object encapsulates:
    - Session configuration and metadata
    - Creation and loading of reproducible analysis sessions
    - Pose preprocessing
    - Spike sorting and LFP preprocessing
    - Video-ephys synchronization
    - Neural alignment to movement data
    - Event based epoching of neural signals

The goal of this module is to provide a consistent, config-based API for multimodal behavioural neuroscience workflows, while preserving reproducibility.

Current implementation
----------------------
The current implementation provides a general-purpose ExperimentSession class designed around the experimental pipelines used during development:
    - Open Ephys acquisition
    - Cambridge Neurotech probes
    - SLEAP-based pose estimation

Future development
------------------
Experiment-specific session subclasses for specialised workflows, for example:
    - ClimbingSession --> Naturalistic Climbing Behaviour
    - LocomotionSession --> Treadmill running/walking
    - OpenFieldSession --> Open field tasks

These classes will extend the base session interface with experiment-specific preprocessing, alignment, and analysis routes.

"""

from pathlib import Path
import shutil
from datetime import datetime
import warnings
from copy import deepcopy

import xmltodict
import pandas as pd
import yaml
import dask.dataframe as dd

# version
from neurokinematics import __version__ as nk_version

from neurokinematics.decorators import log_call

# nk io
from neurokinematics.io import create_session_dirs

# pose
from neurokinematics.pose.preprocessing.base import process_sleap, extract_movement_features
from neurokinematics.pose.io import load_df_list, load_pickle
from neurokinematics.pose.plotting import plot_phase_offset_pairs

# ephys
from neurokinematics.ephys.io import *
from neurokinematics.ephys.spikes.sorting import sort
from neurokinematics.ephys.spikes.rasters import get_movement_aligned_rasters
from neurokinematics.ephys.spikes.plotting import plot_movement_psth, plot_waveforms
from neurokinematics.ephys.lfp.preprocessing import preprocess_lfp
from neurokinematics.ephys.lfp.epochs import get_movement_aligned_erps
from neurokinematics.ephys.lfp.plotting import plot_movement_erps_probe

# multimodal
from neurokinematics.multi_modal.alignment import get_camera_events, align_movements_to_ephys
from neurokinematics.multi_modal.features import get_movement_aligned_features

# models
from neurokinematics.models.registry import MODEL_REGISTRY

# registry
from neurokinematics.registry import EXTRACT_REGISTRY



class ExperimentSession:
    """Class for orchestrating preprocessing and alignment of ephys and pose data

    Example:
        >>> session = ExperimentSession(
        ...     session_id = "demo_session",
        ...     ephys_data_path = "path/containing/ephys/data",
        ...     pose_data_path = "path/containing/pose/data",
        ...     output_root_path = "path/to/store/neurokinematics/session"
        ... )
        >>> session.preprocess_and_align()
    """
    def __init__(self, session_id: str, ephys_data_path: Path | str | None = None, pose_data_path: Path | str | None = None, output_root_path: Path | str | None = None, cfg: str ='demo_session.yaml'):
  
        # set creation date
        self.created_on = datetime.now().isoformat()

        # set session id
        self.session_id = session_id

        # ensure input paths are Path
        if (ephys_data_path == None) & (pose_data_path==None):
            raise ValueError("ExperimentSession requires either a 'pose_data_path', an 'ephys_data_path', or both.")
        
        if isinstance(ephys_data_path, (str, Path)):
            self.ephys_data_path = Path(ephys_data_path)
            if not self.ephys_data_path.exists():
                raise FileNotFoundError(f"Ephys path does not exist: {self.ephys_data_path}")
        else:
            self.ephys_data_path = ephys_data_path

        if isinstance(pose_data_path, (str, Path)):
            self.pose_data_path = Path(pose_data_path)
            if not self.pose_data_path.exists():
                raise FileNotFoundError(f"Pose path does not exist: {self.pose_data_path}")
        else:
            self.pose_data_path = pose_data_path
        
        # then ensure that paths exist...
        # if not self.ephys_data_path.exists():
        #     raise FileNotFoundError(f"Ephys path does not exist: {self.ephys_data_path}")
        # if not self.pose_data_path.exists():
        #     raise FileNotFoundError(f"Pose path does not exist: {self.pose_data_path}")
        
        # load configs
        if cfg is not None:
            self._load_configs(cfg)
            self._set_metadata()
            cfg_output_root_path = self.cfg.get('session', {}).get('output_root', None)
        else:
            self.cfg = {}
            self.metadata = {}
            cfg_output_root_path = None


        # resolve output root
        output_root = output_root_path or cfg_output_root_path
        if output_root is None:
            self.output_root = Path.cwd() / "nk_sessions"
        else:
            self.output_root = Path(output_root)

        # create session directory
        self.session_path = self.output_root / f"{self.session_id}"
        self.dirs = create_session_dirs(self.session_path)

        # create outputs monitor
        self.session_outputs = {}
        self._output_history = {}   # name -> [past provenance records] (kept on overwrite)
        self.session_outputs_path = self.session_path / 'session_outputs.yaml'

        # instantiate blank output file (nested {session_id, outputs} schema)
        if not self.session_outputs_path.exists():
            self._write_session_outputs()


        if cfg is not None:
            self._save_session_config()


    @classmethod
    def from_existing(cls, session_path: Path | str):
        """Loads previously created neurokinematics session

        Args:
            session_path (Path | str): Path to previously created session 

        Returns:
            ExperimentSession: Object replicating the original instatiated session
        """
        session_path = Path(session_path)

        session_config_path = session_path / "session_config.yaml"

        # load previously created config file
        with open(session_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # get runtime config information
        runtime = cfg['session_runtime']

        # setup session
        session = cls(
            session_id = runtime['session_id'],
            ephys_data_path = runtime['ephys_data_path'],
            pose_data_path = runtime['pose_data_path'],
            output_root_path = runtime['output_root'],
            cfg = None
        )

        # set configs 
        session.cfg = cfg['configs']['session']
        session.pose_cfg = cfg['configs']['pose']
        session.lfp_preprocessing_cfg = cfg['configs']['lfp']
        session.multimodal_cfg = cfg['configs']['multimodal']
        session.sorting_cfg = cfg['configs']['spikes']
        session.models_cfg = cfg['configs']['models']

        # get metadata
        session.metadata = cfg.get('session_metadata', {})
        if not session.metadata:
            session._set_metadata()

        # set paths
        session.session_path = session_path

        # load session outputs
        outputs_path = runtime.get("session_outputs_path", "session_outputs.yaml")
        session.session_outputs_path = session_path / outputs_path

        if session.session_outputs_path.exists():
            with open(session.session_outputs_path, "r") as f:
                raw = yaml.safe_load(f)
            session.session_outputs = cls._parse_session_outputs(raw)
            session._output_history = (raw.get('history', {})
                                       if isinstance(raw, dict) else {}) or {}
        else:
            session.session_outputs = {}
            session._output_history = {}
            session._write_session_outputs()

        # recreate dirs
        if 'session_dirs' in runtime:
            session.dirs = {key: Path(val) for key, val in runtime['session_dirs'].items()}
        else:
            session.dirs = create_session_dirs(session.session_path)

        return session

    def __str__(self):
        """
        Returns basic details about the session.
        """
        return "".join(

            [
                "\nExperiment Session Object\n",
                f"\n    Directory: {self.session_path}",
                f"\n    Session ID: {self.session_id}"
            ]
        )
    
    def _load_configs(self, cfg):
        """Loads sub configs listed in the main session config file

        Args:
            cfg (str): Session config file
        """

        self.cfg = load_config(cfg, config_type='session')
        cfg_group = self.cfg['configs'] # dict with the names of the sub configs used in the session
        self.sorting_cfg = load_config(cfg_group['spikes'], config_type='spksorting') # spike sorting config
        self.pose_cfg = load_config(cfg_group['pose'], config_type='pose') # pose config
        self.lfp_preprocessing_cfg = load_config(cfg_group['lfp'], config_type='lfp') # lfp preprocessing config
        self.multimodal_cfg = load_config(cfg_group['multi_modal'], config_type='multimodal') # multimodal alignment config
        self.models_cfg = load_config(cfg_group['models'], config_type = 'models')

    def _save_session_config(self):
        """Freezes session config so session can be loaded at another time
        """

        cfg = {
            'session_runtime':{
                "nk_version": nk_version,
                "session_id": self.session_id,
                "created_on": self.created_on,
                "ephys_data_path": str(self.ephys_data_path) if isinstance(self.ephys_data_path, (str, Path)) else None, # save as str if path, otherwise save as None
                "pose_data_path": str(self.pose_data_path) if isinstance(self.pose_data_path, (str, Path)) else None, # save as str if path, otherwise save as None
                "output_root": str(self.output_root),
                "session_path": str(self.session_path),
                "session_outputs_path": "session_outputs.yaml",
                "session_dirs": {key: str(val) for key, val in self.dirs.items()}
                },
            'configs':{
                'session': self.cfg,
                'pose': self.pose_cfg,
                'spikes': self.sorting_cfg,
                'lfp': self.lfp_preprocessing_cfg,
                'multimodal': self.multimodal_cfg,
                'models': self.models_cfg
                }
            }
        

        session_config_path = self.session_path / "session_config.yaml"

        with open(session_config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    
    def _set_metadata(self):
        """Sets metadata for session - will be expanded in future updates
        """

        self.metadata = {
            'ephys': {
                'acquisition': self.cfg['session']['ephys']['acquisition'],
                'sample_rate': self.sorting_cfg['sample_rate'],
                'lfp_node_idx': self.cfg['session']['ephys']['lfp']['node_idx'],
                'lfp_rec_idx': self.cfg['session']['ephys']['lfp']['rec_idx'],
                'spike_sorter': self.sorting_cfg['sorter']
            },
            'pose':{
                'frame_rate': self.pose_cfg['pose_format']['frame_rate'],
                'tracker': self.pose_cfg['pose_format']['tracker'],
                'node_list': self.pose_cfg['movement_detection']['node_list']
            }
        }

    def _record_session_output(self, file_outputs: dict):#name: str, path: str | Path, file_type: str | None = None, attrs: dict | None = None):
        """Records outputs created during session.

        Args:
            name (str): _description_
            path (str | Path): _description_
            file_type (str | None, optional): _description_. Defaults to None.
            attrs (dict | None, optional): _description_. Defaults to None.
        """
        # provenance shared by every output in this record call
        from neurokinematics.provenance import git_revision
        git_commit  = git_revision()
        config_hash = self._provenance_config_hash()
        run_id      = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not hasattr(self, '_output_history'):
            self._output_history = {}

        for key, val in file_outputs.items():
            name = key
            path = Path(val['path'])
            file_type = val['file_type']
            attrs = val['attrs']

            # need to create a resolver...
            try:
                stored_path = path.relative_to(
                    self.session_path
                )
            except ValueError:
                stored_path = path

            if name in self.session_outputs:
                # archive the previous record (provenance lineage of re-runs)
                self._output_history.setdefault(name, []).append(
                    self.session_outputs[name])

            self.session_outputs[name] = {
                "path": str(stored_path),
                "created": datetime.now().isoformat(),
                "run_id": run_id,             # identifies this processing run
                "nk_version": nk_version,
                "git_commit": git_commit,     # exact code state (+ '-dirty')
                "config_hash": config_hash,   # params that produced this output
                "file_type": file_type,
                "attrs": attrs or {}
            }

        self._write_session_outputs()

    def _write_session_outputs(self):
        """Persist session_outputs.yaml as
        ``{schema_version, session_id, provenance, outputs: {...}}``.

        Output records stay nested under 'outputs' so metadata never leaks into
        the outputs namespace; 'provenance' captures session-level code/input
        versioning, and 'schema_version' lets future readers migrate the format.
        """
        from neurokinematics.provenance import git_revision, SCHEMA_VERSION
        payload = {
            'schema_version': SCHEMA_VERSION,
            'session_id': self.session_id,
            'provenance': {
                'git_commit': git_revision(),
                'nk_version': nk_version,
                'inputs': self._provenance_inputs(),
            },
            'outputs': self.session_outputs,
        }
        # past records of overwritten outputs (omitted when there's no history)
        if getattr(self, '_output_history', None):
            payload['history'] = self._output_history
        with open(self.session_outputs_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    def _provenance_config_hash(self):
        """Hash of all loaded configs — the parameter state for an output."""
        from neurokinematics.provenance import hash_config
        cfgs = {k: getattr(self, k, None) for k in
                ('cfg', 'pose_cfg', 'sorting_cfg', 'lfp_preprocessing_cfg',
                 'multimodal_cfg', 'models_cfg')}
        return hash_config({k: v for k, v in cfgs.items() if v})

    def _provenance_inputs(self):
        """Cheap fingerprints of the session's raw inputs (cached per session)."""
        cached = getattr(self, '_input_fp_cache', None)
        if cached is not None:
            return cached
        from neurokinematics.provenance import fingerprint_input
        fp = {}
        if getattr(self, 'ephys_data_path', None):
            fp['ephys'] = fingerprint_input(self.ephys_data_path)
        if getattr(self, 'pose_data_path', None):
            fp['pose'] = fingerprint_input(self.pose_data_path)
        self._input_fp_cache = fp
        return fp

    @staticmethod
    def _parse_session_outputs(data) -> dict:
        """Extract the output records from loaded session_outputs.yaml content.

        Handles the nested ``{session_id, outputs}`` schema and old flat files,
        and defensively keeps only dict-valued entries so stray metadata keys
        (e.g. a top-level 'session_id') never pollute the outputs dict.
        """
        if not isinstance(data, dict):
            return {}
        if 'outputs' in data and 'session_id' in data:
            raw = data.get('outputs') or {}
        else:
            raw = data
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    

    def _handle_existing_output(self, path: Path, mode: str):
        """Deals with processing/alignment calls if session was already created to avoid accidental overwriting

        Args:
            path (Path): Path to expected data location
            mode (str): Mode to execute, options are 'skip' to skip over function call, 'overwrite' to perform function call, and 'error' to check for accidental overlap

        Raises:
            ValueError: Error if the incorrect mode was selected
            FileExistsError: Error check if the expected output already exists

        Returns:
            bool: Boolean determines whether downstream function call is executed
        """

        if mode not in {"skip", "overwrite", "error"}:
            raise ValueError("Mode must be one of: 'skip', 'overwrite', 'error'")
        
        if not path.exists():
            return True
        
        if mode == "skip":
            return False
        
        if mode == "error":
            raise FileExistsError(f"Output already exists: {path}")
        
        if mode == "overwrite":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True

    def preprocess_and_align(self):
        """Performs all preprocessing and alignment steps
        """
        self.run_pose_processing() # process pose data
        self.run_spike_sorting() # run spike sorting
        self.run_lfp_processing() # process LFP data
        self.align_video() # align video to ephys
        self.align_movements() # align movement events to ephys

    def process(self, type: str, mode: str = "skip"):
        if type == "pose":
            self.run_pose_processing(mode)
        elif type == "spikes":
            self.run_spike_sorting(mode)
        elif type == "lfp":
            self.run_lfp_processing(mode)

    def align(self, type: str, mode: str = 'skip'):
        """Temporally register data streams to the ephys clock.

        'video' -> camera frames, 'movement' -> movement events, 'pose' -> both.
        Neural segmentation lives in :meth:`epoch`, which consumes this output.
        """
        if type == 'video':
            self.align_video(mode)
        elif type == 'movement':
            self.align_movements(mode)
        elif type == 'pose':
            self.align_video(mode)
            self.align_movements(mode)

    def epoch(self, type: str, mode: str = 'skip'):
        """Segment neural data around aligned movement events.

        Runs after :meth:`align` (epoching consumes the movement-event
        alignment). 'spikes' -> movement-aligned rasters, 'lfp' -> ERPs.
        """
        if type == 'spikes':
            self.epoch_spikes(mode)
        elif type == 'lfp':
            self.epoch_lfp(mode)
    
    def extract(self, type: str, feature: str, mode: str = 'skip'):
        
        extract_func = EXTRACT_REGISTRY[type][feature]
        save_path = self.dirs[type] / 'results' / feature
        save_path.mkdir(parents=True, exist_ok=True)

        context = {'dirs': self.dirs}

        if type == 'spikes':
            pass
        elif type == 'lfp':
            pass
        elif type == 'pose':
            pass

    @log_call(label='pose preprocessing', type='run')
    def run_pose_processing(self, mode: str = "skip"):
        """Run preprocessing on markerless pose data and store results

        Args:
            mode (str, optional): Determines whether to perform processing step. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If processing occured and mode is 'skip', returns path to processed data
        """
        if not getattr(self, 'pose_cfg', None):
            raise ValueError(
                "No pose config is loaded for this session. "
                "Recreate the session with a valid session config (e.g. demo_session.yaml)."
            )

        # overwrite utility
        expected_output = self.dirs['pose'] / 'pose_data.csv'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return {"exists": True, "path": expected_output}
        
        self.pose_processed, file_outputs = process_sleap(
            data_path = self.pose_data_path,
            pose_cfg = self.pose_cfg,   # already-loaded config (robust to cfg nesting)
            save_path = self.dirs['pose']
        )

        self._record_session_output(file_outputs)

    # ── Manual inspection (pairs with automated run_qc) ───────────────────────
    # These wrap the same tooling the GUI uses, but as plain session methods so a
    # script/notebook can do: run_qc(subject); session.open_in_phy(); session.inspect_pose()
    # All GUI imports are lazy so importing the data layer never pulls in Qt.

    def phy_output_dir(self):
        """Path to this session's sorter ``phy_output`` folder, or None if unsorted."""
        spikes = (getattr(self, 'dirs', {}) or {}).get('spikes')
        if not spikes:
            return None
        hits = list(Path(spikes).glob('*/phy_output/params.py'))
        return hits[0].parent if hits else None

    def open_in_phy(self, env: str | None = None, gui: str | None = None,
                    conda_exe: str | None = None):
        """Open this session's spike-sorted data in phy2 for manual curation.

        Resolves the sorter's ``phy_output`` folder and launches phy in the
        configured conda environment. *env* / *gui* / *conda_exe* default to the
        GUI's saved phy settings (File ▸ Settings); pass them to override.
        Returns the phy_output path. Raises if the session isn't sorted yet or no
        phy2 environment is configured.
        """
        from neurokinematics.gui.settings import launch_phy, load_settings
        s = load_settings()
        env = env or s.get('phy_env', '')
        if not env:
            raise ValueError(
                "No phy2 environment configured. Pass env=... or set it in the "
                "GUI under File > Settings.")
        phy_dir = self.phy_output_dir()
        if phy_dir is None:
            raise FileNotFoundError(
                "No phy_output found for this session — run spike sorting first.")
        launch_phy(phy_dir, env=env,
                   gui=gui or s.get('phy_gui', 'template-gui'),
                   conda_exe=conda_exe or s.get('conda_exe', 'conda'))
        return phy_dir

    def inspect_pose(self, block: bool = True):
        """Open the interactive pose-quality inspector for this session.

        Shows raw-vs-processed traces and the per-keypoint confidence layout with
        a what-if preview. Creates a Qt application if one isn't already running
        (e.g. from a script or notebook). Returns the dialog.
        """
        from PySide6.QtWidgets import QApplication
        from neurokinematics.gui.pose_inspector import PoseInspectDialog
        existing = QApplication.instance()
        app = existing or QApplication([])
        dlg = PoseInspectDialog(self)
        if existing is not None:
            dlg.exec()          # modal within an already-running app (the GUI)
        else:
            dlg.show()
            if block:
                app.exec()      # standalone: run until the window is closed
        return dlg


    
    @log_call(label='spike sorting', type='run')
    def preprocess_spikes(self):
        """Detect bad channels ahead of sorting — no sorting is performed.

        Reads the ephys recording, applies the probe + a bandpass filter, and runs
        SpikeInterface bad-channel detection. Returns the detection dict (with
        in-memory trace snippets for review) and writes a 'detected' bad-channel
        QC summary into the session's spikes dir. Pair with
        ``run_spike_sorting(bad_channels=...)`` to apply a decision.
        """
        if not getattr(self, 'sorting_cfg', None):
            raise ValueError(
                "No spike-sorting config is loaded for this session. "
                "Recreate the session with a valid session config (e.g. demo_session.yaml)."
            )
        if not getattr(self, 'ephys_data_path', None):
            raise ValueError("No ephys data is linked to this session.")

        from neurokinematics.ephys.spikes.preprocessing import (
            detect_bad_channels, write_bad_channel_report,
        )
        detection = detect_bad_channels(self.ephys_data_path, self.sorting_cfg)
        write_bad_channel_report(self.dirs['spikes'], detection,
                                 removed=[], policy='detected')
        self._spike_detection = detection
        return detection

    def run_spike_sorting(self, mode: str = "skip", bad_channels: list | None = None,
                          preprocess: bool = True):
        """Run spike sorting through SpikeInterface and store results

        Preprocessing (bad-channel detection + QC report) runs first by default,
        so every sort produces bad-channel QC. By default no channels are removed
        (``bad_channels=None`` keeps all); pass a list to drop them, or use the
        GUI's bad-channel review to choose interactively.

        Args:
            mode (str, optional): Determines whether to perform processing step. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".
            bad_channels (list, optional): Channel ids to drop before sorting (e.g. from the bad-channel QC review). None keeps all channels.
            preprocess (bool, optional): Run bad-channel detection first (writes the QC report). Skipped automatically if a caller already ran it for this session. Defaults to True.

        Returns:
            Path: If spike sorting occured and mode is 'skip', returns path to sorting results
        """


        if not getattr(self, 'sorting_cfg', None):
            raise ValueError(
                "No spike-sorting config is loaded for this session. "
                "Recreate the session with a valid session config (e.g. demo_session.yaml)."
            )

        # overwrite utility
        expected_output = self.dirs['spikes'] / 'sorting_analyzer'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return {"exists": True, "path": expected_output}

        # Preprocess first: detect bad channels + write QC report. Skipped if a
        # caller (e.g. the GUI review flow) already detected for this session.
        # A detection hiccup is logged but doesn't block the sort.
        if preprocess and getattr(self, '_spike_detection', None) is None:
            try:
                self.preprocess_spikes()
            except Exception as e:
                print(f"[spike preprocessing] bad-channel detection skipped: {e}")

        # Bad-channel removal is handled inside sort() by the config preprocess
        # pipeline (e.g. 'detect_and_remove_bad_channels'), so we no longer pass a
        # bad_channels list here. The `bad_channels` arg / GUI review is retained as
        # informational QC only (see report below).
        self.sorter, self.recording, self.probe, self.analyzer, self.spike_qc_metrics, file_outputs = sort(
            data_path = self.ephys_data_path,
            cfg_file = self.sorting_cfg,   # already-loaded config (robust to cfg nesting)
            save_path = self.dirs['spikes'],
        )

        # persist a bad-channel QC record alongside the sort
        try:
            from neurokinematics.ephys.spikes.preprocessing import write_bad_channel_report
            detection = getattr(self, '_spike_detection', None)
            if detection is not None:
                cfg = self.sorting_cfg if isinstance(self.sorting_cfg, dict) else {}
                pipeline_removes = 'detect_and_remove_bad_channels' in (cfg.get('preprocess') or {})
                write_bad_channel_report(
                    self.dirs['spikes'], detection,
                    removed=[],   # removal (if any) is done by the preprocess pipeline
                    policy='pipeline' if pipeline_removes else 'keep')
        except Exception:
            pass

        # clear cached detection so a later re-sort detects afresh
        self._spike_detection = None

        self._record_session_output(file_outputs)

    @log_call(label='lfp preprocessing', type='run')
    def run_lfp_processing(self, mode: str = "skip"):
        """Run preprocessing on raw LFP data (chunk, filter, downsample) and store results

        Args:
            mode (str, optional): Determines whether to perform processing step. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If processing occured and mode is 'skip', returns path to processed data
        """

        # overwrite utility
        expected_output = self.dirs['lfp'] / 'lfp_preprocessed'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return {"exists": True, "path": expected_output}

        if self.cfg['session']['ephys']['acquisition'] == 'openephys':
            self.lfp_processed = preprocess_lfp(
                data_path = self.ephys_data_path,
                node_idx = self.metadata['ephys']['lfp_node_idx'],#self.cfg['session']['ephys']['lfp']['node_idx'],
                rec_idx = self.metadata['ephys']['lfp_rec_idx'],#self.cfg['session']['ephys']['lfp']['rec_idx'],
                fs_new = self.lfp_preprocessing_cfg['downsample_rate'],
                chunk_duration_s = self.lfp_preprocessing_cfg['chunking']['chunk_duration_s'],
                pad_duration_s = self.lfp_preprocessing_cfg['chunking']['pad_duration_s'],
                filter_info = {
                    "n_": self.lfp_preprocessing_cfg['filters']["notch"],
                    "bp_": self.lfp_preprocessing_cfg['filters']["bandpass"],
                    "quality": self.lfp_preprocessing_cfg['filters']["quality"]
                },
                dtype = self.lfp_preprocessing_cfg['dtype'],
                save_path = self.dirs['lfp'],
                storage_format = self.lfp_preprocessing_cfg['storage_format']
            )

            # register the output so reloads/GUI know lfp has been processed
            lfp_output_path = Path(self.lfp_processed.output_path)
            self._record_session_output({
                'lfp_data': {
                    'path': str(lfp_output_path),
                    'file_type': lfp_output_path.suffix or self.lfp_processed.storage_format,
                    'attrs': {},
                }
            })

    def align_video(self, mode:str = "skip"):
        """Align frame captures to ephys data

        Args:
            mode (str, optional): Determines whether to perform alignment. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Returns:
            Path: If alignment exists and mode is 'skip', returns path to alignment
        """

        expected_output = self.dirs['alignment'] / 'video_alignment.csv'

        should_run = self._handle_existing_output(expected_output, mode)

        if not should_run:
            return {"exists": True, "path": expected_output}

        _, _, _, _, _ = get_camera_events(
            directory = self.ephys_data_path,
            camera_cfg_file = self.cfg['configs']['multi_modal'],
            save_path = self.dirs['alignment']
        )

    def align_movements(self, mode: str = "skip"):
        """Align movements to ephys data

        Args:
            mode (str, optional): Determines whether to perform alignment. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".

        Raises:
            FileNotFoundError: Checks to make sure the necessary files for running alignment have been created

        Returns:
            Path: If alignment exists and mode is 'skip', returns path to alignment
        """

        # set required files for running alignment that should have been created in the session
        required = [
            self.dirs['pose'] / 'movement_events.pkl',
            self.dirs['pose'] / 'pose_data.csv',
            self.dirs['alignment'] / 'video_alignment.csv'
        ]

        # indicate missing files
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot align movements. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )
        
        # check whether to skip or overwrite alignment
        expected_output = self.dirs['alignment'] / 'movement_event_alignment.csv'
        should_run = self._handle_existing_output(expected_output, mode)
        if not should_run:
            return {"exists": True, "path": expected_output}
        
        # set alignment dirs - 'events' dir is redundant, remove in future updates
        dirs_for_alignment = {
            'events': self.dirs['pose'],
            'pose': self.dirs['pose'],
            'alignment': self.dirs['alignment']
        }

        # run alignment
        self.aligned_movements = align_movements_to_ephys(
            dirs = dirs_for_alignment,
            fs = self.ephys_sample_rate, #self.sorting_cfg['sample_rate'],
            fps = self.pose_sample_rate,   #self.pose_cfg['pose_format']['frame_rate'],
            save_path = self.dirs['alignment']
        )
    
    def epoch_lfp(self, mode: str = "skip"):
        """Epoch lfp data with respect to movement events

        Args:
            mode (str, optional): Determines whether to epoch data. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".
        """

        # set required files for running alignment that should have been created in the session
        required = [
            self.dirs['alignment'] / 'movement_event_alignment.csv',
            self.dirs['lfp'] / 'lfp_preprocessed'
        ]

        # indicate missing files
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot epoch LFP. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )

        # check whether to skip or overwrite epoching
        expected_output = self.dirs['lfp'] / 'lfp_epoched'
        should_run = self._handle_existing_output(expected_output, mode)
        if not should_run:
            return {"exists": True, "path": expected_output}
        
        self.epoch_lfp_root = get_movement_aligned_erps(
            alignment = self.dirs['alignment'] / 'movement_event_alignment.csv',
            lfp_data = self.dirs['lfp'] / 'lfp_preprocessed',
            save_path = self.dirs['lfp'] / 'lfp_epoched'
        )

    def epoch_spikes(self, mode: str = "skip"):
        """Epoch spike rasters with respect to movement events

        Args:
            mode (str, optional): Determines whether to epoch data. Options are 'skip, 'overwrite', or 'error'. 'skip' will be able to run if data isn't present, so reloading session will skip automatically. Use 'overwrite' to re-run. Defaults to "skip".
        """

        # phy export lives under <spikes>/<sorter>/phy_output (spike_times.npy etc.)
        phy_dir = self.dirs['spikes'] / self.spike_sorter / 'phy_output'

        # set required files for running alignment that should have been created in the session
        required = [
            self.dirs['alignment'] / 'movement_event_alignment.csv',
            phy_dir
        ]

        # indicate missing files
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot epoch spikes. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )

        # check whether to skip or overwrite epoching
        expected_output = self.dirs['spikes'] / 'rasters' / 'movement_aligned_rasters.pkl'
        should_run = self._handle_existing_output(expected_output, mode)
        if not should_run:
            return {"exists": True, "path": expected_output}

        self.sorter = load_phy_sorting(phy_dir)
        self.spike_raster_obj = get_movement_aligned_rasters(
            alignment = self.dirs['alignment'] / 'movement_event_alignment.csv',
            sorter = self.sorter,
            save_path = self.dirs['spikes']
        )

    ### * extract features * ###
    def extract_movement_features(self, pre_window_s: float | None = None):
        """Re-extract movement events / features from the saved pose data.

        Regenerates ``movement_events.pkl`` and ``movement_features.zarr`` from
        the already-processed ``pose_data.csv`` without re-running pose cleaning.
        Optionally overrides the pre-movement window (seconds of lead-in before
        each detected onset); when given it is also persisted to the session's
        pose config so it survives a reload.

        Args:
            pre_window_s (float | None, optional): Pre-movement lead-in in seconds.
                ``None`` keeps the value already in the pose config. Defaults to None.
        """
        pose_csv = self.dirs['pose'] / 'pose_data.csv'
        if not pose_csv.exists():
            raise FileNotFoundError(
                f"No processed pose data at {pose_csv}. Run process('pose') first."
            )
        cfg = getattr(self, 'pose_cfg', None)
        if not cfg or 'movement_detection' not in cfg:
            raise ValueError("No pose config with 'movement_detection' loaded for this session.")

        movement_detection = deepcopy(cfg['movement_detection'])
        if pre_window_s is not None:
            movement_detection['pre_window_s'] = float(pre_window_s)
            cfg['movement_detection']['pre_window_s'] = float(pre_window_s)  # persist
            if hasattr(self, '_save_session_config'):
                try:
                    self._save_session_config()
                except Exception as e:
                    warnings.warn(f"pre_window_s not persisted to config: {e}")

        ddf = pd.read_csv(pose_csv)
        extract_movement_features(ddf, movement_detection, self.dirs['pose'])

    def bin_movements_and_spikes(self, bin_size: float = 0.02, return_data: bool = False):

        movement_dataset = self.dirs['pose'] / 'movement_features.zarr'
        alignment = self.dirs['alignment'] / 'video_alignment.csv'
        sorter = self.dirs['spikes'] / 'kilosort4' / 'phy_output'
        
        required = [
            movement_dataset,
            alignment,
            sorter
        ]

        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot extract binned data. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )
        
        save_path = {'pose': self.dirs['pose'], 'spikes': self.dirs['spikes']}
        binned_pose, binned_spikes, unbinned_spikes = get_movement_aligned_features(alignment = alignment, sorter = sorter, movement_dataset = movement_dataset, save_path=save_path, bin_size = bin_size)

        if return_data:
            return binned_pose, binned_spikes, unbinned_spikes
        
    ### * run model * ###
    def fit_unit_model(self, model: str, x_data: str, y_data: str, preset: bool = True, params: dict | None = None):

        if preset:
            params_ = deepcopy(self.models_cfg[model]['preset'])
            if params is not None:
                params = params_ | params # merge params
            else:
                params = params_


        model_fn = MODEL_REGISTRY[model]
        model_fn(x_data, y_data, params, self.dirs['models'])



    ### * plotting * ###
    def plot_spikes(self, unit_ids, plot_params, save_plots: bool = False):

        required = [
            self.dirs['spikes'] / 'movement_aligned_rasters.pkl'
        ] 

        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot plot spikes. Missing required files:\n"
                + "\n".join(str(p) for p in missing)
            )
        
        rasters_df = load_pickle(self.dirs['spikes'] / 'movement_aligned_rasters.pkl', method = 'pandas')

        if save_plots:
            plot_movement_psth(rasters_df, unit_ids, plot_params, self.dirs['plots'])
        else:
            plot_movement_psth(rasters_df, unit_ids, plot_params)
    
    @property
    def ephys_sample_rate(self):
        """ Returns sample rate of ephys acquisition
        """
        return self.metadata['ephys']['sample_rate']
    
    @property
    def pose_sample_rate(self):
        """ Returns frame rate of original pose data
        """
        return self.metadata['pose']['frame_rate']
    
    @property
    def acquisition_system(self):
        """ Returns name of ephys acquisition system
        """
        return self.metadata['ephys']['acquisition']
    
    @property
    def pose_package(self):
        """ Returns name of package used for pose data 
        """
        return self.metadata['pose']['tracker']
    
    @property
    def spike_sorter(self):
        """ Returns name of spike sorter used in session
        """
        return self.metadata['ephys']['spike_sorter']

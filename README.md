[![Unit Test](https://github.com/cjblack/neurokinematics/actions/workflows/session_class_test.yml/badge.svg)](https://github.com/cjblack/neurokinematics/actions/workflows/session_class_test.yml)

 # neurokinematics

 ## Overview

 `neurokinematics` is a Python package for organising and processing electrophysiology (neuro) and markerless pose estimation (kinematics) data from behavioural neuroscience experiments.

 The project focuses on providing a consistent interface for preparing neural and behavioural datasets for downstream analysis, with particular emphasis on aligning movement events with extracellular recordings.

 It currently supports workflows for:
 - Preprocessing SLEAP-based pose data and extracting movement events.
 - Spike sorting and basic analysis through SpikeInterface.
 - Preprocessing and epoching local field potential (LFP) data.
 - Aligning behavioural and neural data streams.

 This codebase has been developed alongside experimental work on ethologically relevant behaviours in mice ([Naturalistic climbing reveals adaptive strategies for interlimb coordination in freely moving mice](https://doi.org/10.1016/j.isci.2026.115901)), and is designed around a specific experimental setup (Open Ephys acquisition, Cambridge Neurotech probes, and SLEAP pose estimation). 
 
 Some components are currently tailored to this setup and require experiment-specific configuration and validation. However, ongoing development is focused on generalising these workflows to support a wider range of data formats and acquisition systems.

## Features
- Centralised preprocessing of electrophysiology and markerless pose data.
- Tools for extracting and structuring movement and neural signals for downstream analysis.
- Modular subpackages (pose, ephys, multi_modal) that can be used independently.
- Config-driven workflows to support reproducible processing across sessions.

 ## Quick install

 Clone repository, and create environment using `environment_cuda.yml`:
 
 ```bash
git clone https://github.com/cjblack/neurokinematics.git
cd neurokinematics
conda env create -f environment_cuda.yml --solver=libmamba
conda activate neurokinematics_cuda
 ```

This environment installs PyTorch with CUDA support for GPU accelerated spike-sorting.

### &#x26a0;&#xfe0f; Requirements
- NVIDIA GPU with CUDA support
- Compatible drives and CUDA toolkit installed
- Tested on:
    - NVIDIA RTX 50 series
    - NVIDIA RTX 20 series

## Repo structure
```text
neurokinematics/
├── neurokinematics/            # Source package
│   ├── pose/                   # Pose processing
│   ├── ephys/ 
│   │   ├──channel_maps/        # Custom acquisition channel maps
│   │   ├──lfp/                 # LFP processing
│   │   ├──probe_interfaces/    # Probe geometry/metadata
│   │   └──spikes/              # Spike processing
│   ├── multi_modal/            # Cross-modal alignment
│   └── data/                   # Session/data abstractions
├── configs/                    # YAML configuration files
├── docs/                       # Extended documentation/example figures
├── tests/                      # Unit and integration tests
├── examples/                   # Example workflow scripts
└── notebooks/                  # Jupyter demos
```

## Documentation
- [Pose processing](docs/pose.md)
- [Spike sorting](docs/spikes.md)
- [LFP processing](docs/lfp.md)
- [Multi-modal](docs/multi_modal.md)
- [Data](docs/data.md)

## Examples

Example scripts are provided in `examples/`, and example datasets are published on OSF: https://doi.org/10.17605/OSF.IO/3SR67

**Not all data is required for example scripts.**


### Quick start (minimal pose-only)
Only the sample pose data from the OSF repository is required for this example.
- `pose_data` ~3.3MB

Run:

```bash
python examples/run_pose_example.py
```
This example shows:
- Pose preprocessing
- Movement event extraction

#### Reload
Requires `run_pose_example.py` to be called first. Similarly only uses pose data.

Run:

```bash
python examples/run_load_existing_session.py
```
This example shows:
- Loading a previously created session
- Log call for re-running analysis

### Full multi-modal

This example reproduces the full processing pipeline used during development and requires both pose and ephys datasets.
- `pose_data` ~3.3MB
- `ephys_data` ~8.5GB

> **Note:** the electrophysiology dataset is distributed as individual files on OSF.
> If any files appear to be missing, corrputed, or fail during testing,
> please open an issue on
> https://github.com/cjblack/neurokinematics/issues
> or leave a comment on the OSF project page

Run:

```bash
python examples/run_session_pipeline.py
```

This example shows:
- Pose preprocessing
- Spike sorting
- LFP preprocessing
- Pose processing
- Video-ephys alignment
- Movement-ephys alignment

## Scope and current support

This package has been developed and tested primarily on a specific experimental setup:
- SLEAP-based pose estimation
- Open Ephys acquisition system
- Cambridge Neurotech probes (specifically 64-channel H5 probes)

The core components are designed to be modular and extensible with the goal of supporting a wider range of data formats and recording systems.

At present, some assumptions about input structure (e.g. alignment data, config formats) reflect the original datasets used in development. These are documented in the relevant modules and can be adapted with minimal changes.

Future work will focus on generalising input interfaces and expanding format support including:
- Pose packages (e.g. DLC, Anipose)
- Ephys acquisition systems
- Probes (e.g. Neuropixels, Neuralynx)
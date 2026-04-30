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

## Documentation
- [Pose processing](docs/pose.md)
- [Spike sorting](docs/spikes.md)
- [LFP processing](docs/lfp.md)
- [Multi-modal](docs/multi_modal.md)
- [Data](docs/data.md)


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
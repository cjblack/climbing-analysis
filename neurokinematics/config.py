from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class _Config(BaseModel):
    """Shared base for all config schemas.

    ``extra='forbid'`` rejects any key that isn't declared on the schema, so a
    typo (e.g. ``samplerate:`` instead of ``sample_rate:``) fails loudly at load
    time instead of being silently dropped.
    """
    model_config = ConfigDict(extra="forbid")


class SpikeSortingConfig(_Config):
    # required
    name: str
    probe_manufacturer: str
    probe_id: str
    sample_rate: float

    # defaulted
    rec_type: Literal["openephys", "spikeglx", "intan"] = "openephys"
    sorter: str = "kilosort4"

    # optional
    channel_map: Optional[str] = None
    group_mode: Optional[str] = None
    stream_name: Optional[str] = None
    rec_node: Optional[int] = None

    # nested
    to_compute: dict = Field(default_factory=dict)
    quality_metrics: list[str] = Field(default_factory=list)
    curation: dict = Field(default_factory=dict)
    preprocess: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _stream_and_probe_consistency(self):
        npx = (self.probe_manufacturer.lower() == "imec" and self.probe_id.lower() == "neuropixels")
        if npx:
            if self.stream_name:
                raise ValueError(
                    "SpikeInterface auto-detects Neuropixels stream name. "
                    "Remove 'stream_name' and set 'rec_node' instead."
                )
            if self.rec_node is None:
                raise ValueError("Neuropixels requires 'rec_node' (Open Ephys Record Node ID).")
        else:
            if not self.stream_name and self.rec_node is None:
                raise ValueError("Set 'stream_name' or 'rec_node' so the recording stream can be resolved.")
        return self


class LFPConfig(_Config):
    name: str
    dtype: str
    chunking: dict = Field(default_factory=dict)
    filters: dict = Field(default_factory=dict)
    downsample_rate: float
    storage_format: str


class PoseConfig(_Config):
    pose_format: dict = Field(default_factory=dict)
    pose_preprocessing: dict = Field(default_factory=dict)
    post_processing: dict = Field(default_factory=dict)
    movement_detection: dict = Field(default_factory=dict)


class MultiModalConfig(_Config):
    name: str
    camera: dict = Field(default_factory=dict)
    acquisition_settings: dict = Field(default_factory=dict)
    detection_settings: dict = Field(default_factory=dict)


class ModelConfig(_Config):
    glm: dict = Field(default_factory=dict)


class SessionConfig(_Config):
    session: dict = Field(default_factory=dict)
    configs: dict = Field(default_factory=dict)
    pipeline: dict = Field(default_factory=dict)


# Registry must come *after* the classes it references (the module runs top-to-bottom).
REGISTRY = {
    "spksorting": SpikeSortingConfig,
    "pose":       PoseConfig,
    "lfp":        LFPConfig,
    "multimodal": MultiModalConfig,
    "models":     ModelConfig,
    "session":    SessionConfig,
}

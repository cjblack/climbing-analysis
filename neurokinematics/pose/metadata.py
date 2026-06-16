from pathlib import Path
import re

def resolve_file_metadata(filename, meta_cfg):
    source = meta_cfg.get('source', 'filename')

    return _METADATA_RESOLVERS[source](filename, meta_cfg)



def _from_filename(filename, meta_cfg):
    name = Path(filename).name
    m = re.match(meta_cfg['pattern'], name)
    if not m:
        raise ValueError(
            f"Filename {name!r} did not match pattern "
            f"{meta_cfg['pattern']!r}. Check pose_format.metadata in config to reformat pattern."
        )
    meta = m.groupdict()
    return meta

def _from_manifest(filename, meta_cfg):
    pass

def _from_sidecar(filename, meta_cfg):
    pass

def _from_custom(filename, meta_cfg):
    """Add your own custom layout for parsing your metadata

    Ensure that metadata returns a dictionary containing
    {
        'Id': str, # subject id
        'Type': str, # experiment type/name
        'Date': datetime, # date experiment was performed
        'Trial': int # id of trial - this is used for sorting/querying dataframes later on
    }
    
    """
    pass

_METADATA_RESOLVERS = {
    'filename': _from_filename,
    'manifest': _from_manifest,
    'sidecar': _from_sidecar,
    'custom': _from_custom
}
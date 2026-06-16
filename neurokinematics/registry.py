import importlib, inspect, pkgutil, warnings

# Load state

_EXTRACTORS_LOADED = False


# EXRACTORS #
EXTRACTOR_PACKAGES = (
    'neurokinematics.pose.extractors',
)
EXTRACT_REGISTRY = {}
_REQUIRED = {'data', 'params', 'save_path'}

def register_extractor(type, feature):
    def deco(fn):
        missing = _REQUIRED - set(inspect.signature(fn).parameters)
        if missing:
            raise TypeError(
                f"Extractor '{type}/{feature}' ({fn.__name__} is missing) "
                f"parameter(s): {sorter(missing)}"
            )
        EXTRACT_REGISTRY.setdefault(type, {})[feature] = fn
        return fn
    return deco


def load_extractors(force = False):
    global _EXTRACTORS_LOADED
    if _EXTRACTORS_LOADED and not force:
        return
    for pkg_name in EXTRACTOR_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            try:
                importlib.import_module(f"{pkg_name}.{mod.name}")
            except Exception as e:
                warnings.warn(f"Could not load extractor '{pkg_name}.{mod.name}': {e}")
    _EXTRACTORS_LOADED = True


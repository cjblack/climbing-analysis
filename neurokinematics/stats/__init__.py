# bayesian imports
from .bayesian.hierarchical import fit_hierarchical_linear


MODEL_REGISTRY = {
    'bayesian':{
        'hierarchical_linear': fit_hierarchical_linear,
    },
    'frequentist':{

    }
}   


def get_model(framework: str, model: str):
    if framework not in MODEL_REGISTRY:
        raise ValueError(f"Framework {framework} not recognised." 
                         "Choose from {list(MODEL_REGISTRY.keys())}")
    if model not in MODEL_REGISTRY[framework]:
        raise ValueError(f"Model {model} not recognised." 
                         f"Choose from {framework} models: {list(MODEL_REGISTRY[framework].keys())}")
    
    return MODEL_REGISTRY[framework][model]
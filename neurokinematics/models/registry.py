from .glm import create_glm, compare_glm_models



MODEL_REGISTRY = {
    'glm': compare_glm_models,
    #'vae': create_vae,
    #'pca': create_pca,
    #'lfads': create_lfads,
}
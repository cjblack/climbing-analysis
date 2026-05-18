from .glm import create_glm



MODEL_REGISTRY = {
    'glm': create_glm,
    #'vae': create_vae,
    #'pca': create_pca,
    #'lfads': create_lfads,
}
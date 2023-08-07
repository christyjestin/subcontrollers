import numpy as np

def cosine_similarity(a, b):
    assert len(a.shape) == 1 and a.shape == b.shape, "Both inputs must be 1-d arrays of the same size"
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
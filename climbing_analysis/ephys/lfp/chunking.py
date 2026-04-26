import numpy as np

def iter_chunks(n_samples: int, chunk_size, pad_size):
    """Creates iterable for chunking data

    Args:
        n_samples (int): Number of samples in data.
        chunk_size (_type_): Size of chunk for data.
        pad_size (_type_): Size of padding for chunk.

    Yields:
        _type_: _description_
    """

    for core_start in range(0, n_samples, chunk_size):
        core_end = min(core_start + chunk_size, n_samples)

        read_start = max(0, core_start - pad_size)
        read_end = min(n_samples, core_end + pad_size)

        yield read_start, read_end, core_start, core_end
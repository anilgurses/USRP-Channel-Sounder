import numpy as np
from sigmf import sigmffile

def read_sigmf_file(filepath_meta):
    """
    Reads a SigMF file given the path to its .sigmf-meta file.

    Args:
        filepath_meta (str): The absolute path to the .sigmf-meta file.

    Returns:
        tuple: A tuple containing the SigMFFile object and the samples.
               Returns (None, None) if an error occurs.
    """
    try:
        signal = sigmffile.fromfile(filepath_meta)
        samples = signal.read_samples()
        return signal, samples
    except Exception as e:
        print(f"Error reading SigMF file {filepath_meta}: {e}")
        return None, None

def read_sigmf_data_direct(filepath_data, dtype=np.complex64):
    """
    Reads a .sigmf-data file directly using numpy.

    Args:
        filepath_data (str): The absolute path to the .sigmf-data file.
        dtype (numpy.dtype): The data type of the samples (e.g., np.complex64).

    Returns:
        numpy.ndarray: The samples read from the file.
                       Returns None if an error occurs.
    """
    try:
        samples = np.fromfile(filepath_data, dtype=dtype)
        return samples
    except Exception as e:
        print(f"Error reading SigMF data file {filepath_data}: {e}")
        return None

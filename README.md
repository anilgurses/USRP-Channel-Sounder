# Channel Sounder

This is a real-time channel sounder with different waveform options. It can easily be configured through the config files located under configs directory. 

## Getting Started

There are 2 options to install channel sounder,

### With Docker

```
# Network Host option can be ignored
$ sudo docker build --network=host -t ch-sounder .
$ sudo docker run -dit --network=host --privileged -v /dev:/dev --name sounder ch-sounder
$ sudo docker exec -it sounder bash
$ cd sounder
$ python3 main.py -c ../config/tx_config.yaml 
```

### Without Docker

If you choose not to use Docker, you can do it by installing all the dependencies. 
 
```
$ sudo apt-get update -y 
$ sudo apt-get install libuhd-dev uhd-host
$ uhd_images_downloader
$ python3 -m pip install -r requirements.txt
$ cd sounder
$ python3 main.py -c ../config/tx_config.yaml 
```

Dataset link: https://datadryad.org/dataset/doi:10.5061/dryad.7h44j105p

## USAGE 

### Running the Channel Sounder

To run the channel sounder, navigate to the `sounder` directory and execute `main.py` with a configuration file:

```bash
cd sounder
python3 main.py -c ../config/tx_config.yaml 
```

Replace `../config/tx_config.yaml` with the appropriate configuration file for your use case (e.g., `rx_config.yaml`).

## Post Processing

The `post_processing` directory contains several Jupyter notebooks for analyzing and visualizing the collected data. To run these notebooks, you'll need to install the dependencies listed in `post_processing/requirements.txt`:

```bash
pip install -r post_processing/requirements.txt
```

### Notebooks

*   **Antenna.ipynb**: Visualizes antenna radiation patterns.
*   **PostProcess.ipynb**: Performs post-processing on the collected data, including channel estimation and path loss analysis.
*   **RX_review.ipynb**: Reviews and analyzes the received signal data.
*   **SigMF_conv.ipynb**: Converts the data to SigMF format.
*   **SigMF_demo.ipynb**: Demonstrates how to read and work with SigMF files.

#### Improviser's Path: Reading SigMF Data Files

The dataset provided (link above) contains SigMF formatted files. If you prefer to not use my post processing script and work on raw I/Q data, you can do so by reading the raw I/Q data files directly. To read these files in Python, you'll need the `sigmf` and `numpy` libraries. Install them using pip:

```bash
pip install sigmf numpy
```

Then, you can use the following Python code to load the signal data:

```python
import numpy as np
from sigmf import SigMFFile, sigmffile

# Create a SigMFFile object from a .sigmf-meta file
# Replace 'path/to/your/file.sigmf-meta' with the actual path to your metadata file
signal = sigmffile.fromfile('path/to/your/file.sigmf-meta')

# Read the samples
samples = signal.read_samples()

# Alternatively, you can read the .sigmf-data file directly with numpy
# Replace 'path/to/your/file.sigmf-data' with the actual path to your data file
samples_direct = np.fromfile('path/to/your/file.sigmf-data', dtype=np.complex64) # Assuming complex64 data
```

## Cite 

If you use this channel sounder or the associated dataset in your research, please cite the following:




```
@inproceedings{gurses2024air,
  author    = {A. G{\"u}rses and M. L. Sichitiu},
  title     = {Air-to-Ground Channel Modeling for UAVs in Rural Areas},
  booktitle = {2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)},
  year      = {2024},
  pages     = {1--6},
  address   = {Washington, DC, USA},
  doi       = {10.1109/VTC2024-Fall63153.2024.10757825},
  publisher = {IEEE}
}
```